mod distribution;
mod ga;
mod game;
mod plotting;

use chrono::Local;
use csv::Writer;
use mimalloc::MiMalloc;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use statrs::statistics::Statistics;
use std::env;
use std::fs;
use std::io::{self, Write};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Clone, Debug)]
struct RunningStats {
    n: usize,
    mean: f64,
    m2: f64,
    min: f64,
    max: f64,
}

#[derive(Clone, Debug)]
struct RunConfig {
    worker_count: usize,
}

impl RunningStats {
    fn new() -> Self {
        Self {
            n: 0,
            mean: 0.0,
            m2: 0.0,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
        }
    }

    fn update(&mut self, x: f64) {
        self.n += 1;
        let n_f = self.n as f64;
        let delta = x - self.mean;
        self.mean += delta / n_f;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;

        if x < self.min {
            self.min = x;
        }
        if x > self.max {
            self.max = x;
        }
    }

    fn std_dev(&self) -> f64 {
        if self.n > 1 {
            (self.m2 / (self.n as f64 - 1.0)).sqrt()
        } else {
            0.0
        }
    }
}

fn main() {
    let run_config = parse_run_config();
    let worker_count = run_config.worker_count;
    ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build_global()
        .unwrap_or_else(|e| panic!("Failed to configure Rayon thread pool: {}", e));

    let start = std::time::Instant::now();

    // Create output directory with timestamp
    let timestamp = Local::now().format("%Y-%m-%d %H-%M-%S").to_string();
    let output_dir = format!("./output/{}", timestamp);
    fs::create_dir_all(format!("{}/figure-1", output_dir)).expect("Failed to create figure-1 dir");
    fs::create_dir_all(format!("{}/figure-2", output_dir)).expect("Failed to create figure-2 dir");
    fs::create_dir_all("./output").expect("Failed to create output dir");

    let filename = format!("{}/result.csv", output_dir);
    let global_filename = "./output/result.csv".to_string();

    // Create CSV headers
    let headers = vec![
        "T", "c_t", "k", "f", "E_D", "E_P", "E_V", "E_cf", "Var_D", "Var_P",
        "Var_V", "Var_cf", "m11", "lambda1", "theta1", "m12",
        "lambda2", "theta2", "m21", "gamma1", "mu1", "m22", "gamma2", "mu2", "q", "r", "p1", "p2",
        "cons_1", "cons_2", "cons_3", "pi",
    ];

    // Market parameters
    let e_d = 100.0;
    let e_p = 1400.0;
    let e_v = 100.0;

    let var_d = 3.0;
    let var_p = 2.0;
    let var_v = 2.0;

    let (mu_d, sigma_d) = distribution::lognormal_params_from_mean_var(e_d, var_d);
    let (mu_p, sigma_p) = distribution::lognormal_params_from_mean_var(e_p, var_p);
    let (mu_v, sigma_v) = distribution::lognormal_params_from_mean_var(e_v, var_v);

    let rho_dp = 0.5;
    let rho_pv = 0.3;
    let rho_dv = 0.2;

    let e_cf = 60.0;
    let var_cf = 10.0;

    // Build covariance matrix
    let mean = vec![mu_d, mu_p, mu_v];
    let cov = vec![
        vec![
            sigma_d * sigma_d,
            rho_dp * sigma_d * sigma_p,
            rho_dv * sigma_d * sigma_v,
        ],
        vec![
            rho_dp * sigma_d * sigma_p,
            sigma_p * sigma_p,
            rho_pv * sigma_p * sigma_v,
        ],
        vec![
            rho_dv * sigma_d * sigma_v,
            rho_pv * sigma_p * sigma_v,
            sigma_v * sigma_v,
        ],
    ];

    let total_iterations = 100usize;
    println!(
        "Starting RWA simulation with {} iterations...",
        total_iterations
    );

    // Shared CSV writers (timestamped + global), write header once
    let writers = Arc::new(Mutex::new((
        Writer::from_path(&filename).expect("Failed to open timestamped CSV"),
        Writer::from_path(&global_filename).expect("Failed to open global CSV"),
    )));
    {
        let mut guard = writers.lock().expect("Failed to lock writers");
        guard
            .0
            .write_record(&headers)
            .expect("Failed to write timestamped CSV header");
        guard
            .1
            .write_record(&headers)
            .expect("Failed to write global CSV header");
    }

    // Running stats (no full in-memory results)
    let stats = Arc::new(Mutex::new(vec![RunningStats::new(); headers.len()]));

    // Progress
    let thread_status = Arc::new(
        (0..worker_count)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );
    let completed_count = Arc::new(AtomicUsize::new(0));
    let stop_progress = Arc::new(AtomicBool::new(false));

    let skipped_figure_count = Arc::new(AtomicUsize::new(0));
    let csv_error_count = Arc::new(AtomicUsize::new(0));
    let plot_error_count = Arc::new(AtomicUsize::new(0));

    let progress_lines = worker_count + 1;
    println!("Overall progress: 0/{}", total_iterations);
    for tid in 0..worker_count {
        println!("Thread {:02}: idle", tid);
    }

    let reporter_status = Arc::clone(&thread_status);
    let reporter_completed = Arc::clone(&completed_count);
    let reporter_stop = Arc::clone(&stop_progress);
    let reporter = Some(thread::spawn(move || {
        let mut stdout = io::stdout();

        loop {
            print!("\x1B[{}A", progress_lines);

            let done = reporter_completed.load(Ordering::Relaxed);
            print!("\x1B[2K\rOverall progress: {}/{}\n", done, total_iterations);

            for tid in 0..worker_count {
                let current = reporter_status[tid].load(Ordering::Relaxed);
                if current == 0 {
                    print!("\x1B[2K\rThread {:02}: idle\n", tid);
                } else {
                    print!(
                        "\x1B[2K\rThread {:02}: processing iteration {}/{}\n",
                        tid, current, total_iterations
                    );
                }
            }

            let _ = stdout.flush();

            if reporter_stop.load(Ordering::Relaxed) {
                break;
            }

            thread::sleep(Duration::from_millis(120));
        }
    }));

    let worker_status = Arc::clone(&thread_status);
    let worker_completed = Arc::clone(&completed_count);
    let worker_writers = Arc::clone(&writers);
    let worker_stats = Arc::clone(&stats);
    let worker_skipped = Arc::clone(&skipped_figure_count);
    let worker_csv_err = Arc::clone(&csv_error_count);
    let worker_plot_err = Arc::clone(&plot_error_count);

    // Stream processing: compute -> write -> plot, per iteration
    let run_iteration = |i: usize, thread_id: Option<usize>| {
        if let Some(tid) = thread_id {
            worker_status[tid].store(i + 1, Ordering::Relaxed);
        }

        let (result_data, price_sample, cost_sample) = process_iteration(e_cf, var_cf, &mean, &cov);

        // Update running stats
        if let Ok(mut s) = worker_stats.lock() {
            for (j, value) in result_data.iter().enumerate() {
                if let Some(col) = s.get_mut(j) {
                    col.update(*value);
                }
            }
        }

        // Write CSV row immediately and flush to disk so data survives interrupts
        let row_str: Vec<String> = result_data.iter().map(|v| v.to_string()).collect();
        match worker_writers.lock() {
            Ok(mut w) => {
                let write_ok = w.0.write_record(&row_str).is_ok()
                    && w.0.flush().is_ok()
                    && w.1.write_record(&row_str).is_ok()
                    && w.1.flush().is_ok();
                if !write_ok {
                    worker_csv_err.fetch_add(1, Ordering::Relaxed);
                }
            }
            Err(_) => {
                worker_csv_err.fetch_add(1, Ordering::Relaxed);
            }
        }

        // Plot immediately
        let q = result_data[28];
        let r = result_data[29];
        let p1 = result_data[30];
        let p2 = result_data[31];

        if !(q.is_finite() && r.is_finite() && p1.is_finite() && p2.is_finite()) {
            worker_skipped.fetch_add(1, Ordering::Relaxed);
        } else if plotting::save_iteration_figures(
            i,
            &price_sample,
            &cost_sample,
            mu_p,
            sigma_p,
            e_cf,
            var_cf,
            &output_dir,
        )
        .is_err()
        {
            worker_plot_err.fetch_add(1, Ordering::Relaxed);
        }

        worker_completed.fetch_add(1, Ordering::Relaxed);
    };

    (0..total_iterations)
        .into_par_iter()
        .for_each(|i| run_iteration(i, rayon::current_thread_index()));

    stop_progress.store(true, Ordering::Relaxed);
    if let Some(handle) = reporter {
        let _ = handle.join();
        println!();
    }

    // Flush CSV
    if let Ok(mut w) = writers.lock() {
        if w.0.flush().is_err() || w.1.flush().is_err() {
            println!("Warning: failed to flush CSV writers.");
        }
    }

    println!("Global results also saved to {}", global_filename);
    println!("Figure 1 files saved to {}/figure-1", output_dir);
    println!("Figure 2 files saved to {}/figure-2", output_dir);
    println!(
        "Skipped figure generation for {} iterations due to unsolved q/r/p1/p2",
        skipped_figure_count.load(Ordering::Relaxed)
    );

    let csv_err = csv_error_count.load(Ordering::Relaxed);
    if csv_err > 0 {
        println!("Warning: {} iterations failed to write CSV rows.", csv_err);
    }

    let plot_err = plot_error_count.load(Ordering::Relaxed);
    if plot_err > 0 {
        println!(
            "Warning: {} iterations failed to generate figures.",
            plot_err
        );
    }

    // Print summary statistics from running stats
    if let Ok(s) = stats.lock() {
        print_summary_statistics(&s, &headers, total_iterations);
    }

    println!("Results saved to {}", filename);
    println!("Total time: {:?}", start.elapsed());
}

fn parse_run_config() -> RunConfig {
    let mut args = env::args().skip(1);
    let mut worker_count: Option<usize> = None;

    while let Some(arg) = args.next() {
        if arg == "--workers" || arg == "-w" {
            let value = args.next().unwrap_or_else(|| {
                eprintln!("Missing value for {}. Example: --workers 4", arg);
                std::process::exit(1);
            });

            let workers = value.parse::<usize>().unwrap_or_else(|_| {
                eprintln!(
                    "Invalid worker count '{}'. Please provide a positive integer.",
                    value
                );
                std::process::exit(1);
            });

            if workers == 0 {
                eprintln!("Worker count must be greater than 0.");
                std::process::exit(1);
            }
            worker_count = Some(workers);
        }
    }

    if let Some(workers) = worker_count {
        println!("Using manually configured worker count: {}", workers);
        return RunConfig {
            worker_count: workers,
        };
    }

    let default_workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    println!(
        "No --workers specified, using logical processor count: {}",
        default_workers
    );
    RunConfig {
        worker_count: default_workers,
    }
}

fn print_summary_statistics(stats: &[RunningStats], headers: &[&str], total_iterations: usize) {
    if stats.is_empty() {
        return;
    }

    println!("\n=== Summary Statistics ===");
    println!("Total iterations: {}", total_iterations);

    for (i, &header) in headers.iter().enumerate() {
        if let Some(col) = stats.get(i) {
            println!(
                "{}: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
                header,
                col.mean,
                col.std_dev(),
                col.min,
                col.max
            );
        }
    }
    println!("========================\n");
}

fn process_iteration(
    e_cf: f64,
    var_cf: f64,
    mean: &[f64],
    cov: &[Vec<f64>],
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    // Generate samples
    let cf_samples = distribution::operation_cost_gamma(100, e_cf, var_cf, None);
    let (d_samples, p_samples, v_samples) =
        distribution::sample_multivariate_lognormal(100, mean, cov, None);

    // Run game simulation
    let res = game::start_game(100, &d_samples, &p_samples, &v_samples, &cf_samples);

    let mean_d = d_samples.clone().mean();
    let mean_p = p_samples.clone().mean();
    let mean_v = v_samples.clone().mean();
    let mean_cf = cf_samples.clone().mean();
    let var_d = d_samples.clone().variance();
    let var_p = p_samples.clone().variance();
    let var_v = v_samples.clone().variance();
    let var_cf = cf_samples.clone().variance();

    let args = [
        game::T,
        game::C_T,
        game::K,
        game::F,
        mean_d,
        mean_p,
        mean_v,
        mean_cf,
        var_d,
        var_p,
        var_v,
        var_cf,
    ];

    let result_data = [&args[..], &res[..]].concat();

    (result_data, p_samples, cf_samples)
}
