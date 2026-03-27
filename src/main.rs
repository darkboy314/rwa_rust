mod distribution;
mod ga;
mod game;
mod plotting;
mod stats;

use chrono::Local;
use csv::Writer;
use mimalloc::MiMalloc;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use stats::{mean as sample_mean, variance as sample_variance};
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

struct OutputPaths {
    output_dir: String,
    timestamped_csv: String,
    global_csv: String,
}

struct SimulationParams {
    headers: Vec<&'static str>,
    mean: Vec<f64>,
    cov: Vec<Vec<f64>>,
    mu_p: f64,
    sigma_p: f64,
    e_cf: f64,
    var_cf: f64,
    total_iterations: usize,
}

struct SharedState {
    writers: Arc<Mutex<(Writer<std::fs::File>, Writer<std::fs::File>)>>,
    stats: Arc<Mutex<Vec<RunningStats>>>,
    thread_status: Arc<Vec<AtomicUsize>>,
    completed_count: Arc<AtomicUsize>,
    stop_progress: Arc<AtomicBool>,
    skipped_figure_count: Arc<AtomicUsize>,
    csv_error_count: Arc<AtomicUsize>,
    plot_error_count: Arc<AtomicUsize>,
}

struct FinalCounts {
    skipped_figures: usize,
    csv_errors: usize,
    plot_errors: usize,
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
    configure_thread_pool(run_config.worker_count);

    let start = std::time::Instant::now();
    let output_paths = create_output_paths();
    let simulation = build_simulation_params();
    let total_iterations = simulation.total_iterations;
    println!(
        "Starting RWA simulation with {} iterations...",
        total_iterations
    );
    let shared =
        initialize_shared_state(&output_paths, &simulation.headers, run_config.worker_count);
    let reporter = start_progress_reporter(
        Arc::clone(&shared.thread_status),
        Arc::clone(&shared.completed_count),
        Arc::clone(&shared.stop_progress),
        run_config.worker_count,
        total_iterations,
    );

    run_simulation(&simulation, &output_paths, &shared);

    stop_progress_reporter(shared.stop_progress.as_ref(), reporter);
    flush_writers(&shared.writers);

    let final_counts = collect_final_counts(&shared);
    print_run_summary(
        &output_paths,
        &simulation.headers,
        total_iterations,
        &shared.stats,
        &final_counts,
        start.elapsed(),
    );
}

fn configure_thread_pool(worker_count: usize) {
    ThreadPoolBuilder::new()
        .num_threads(worker_count)
        .build_global()
        .unwrap_or_else(|e| panic!("Failed to configure Rayon thread pool: {}", e));
}

fn create_output_paths() -> OutputPaths {
    let timestamp = Local::now().format("%Y-%m-%d %H-%M-%S").to_string();
    let output_dir = format!("./output/{}", timestamp);
    fs::create_dir_all("./output").expect("Failed to create output dir");
    fs::create_dir_all(format!("{}/figure-1", output_dir)).expect("Failed to create figure-1 dir");
    fs::create_dir_all(format!("{}/figure-2", output_dir)).expect("Failed to create figure-2 dir");

    OutputPaths {
        timestamped_csv: format!("{}/result.csv", output_dir),
        global_csv: "./output/result.csv".to_string(),
        output_dir,
    }
}

fn build_simulation_params() -> SimulationParams {
    let headers = vec![
        "T", "c_t", "k", "f", "E_D", "E_P", "E_V", "E_cf", "Var_D", "Var_P", "Var_V", "Var_cf",
        "m11", "lambda1", "theta1", "m12", "lambda2", "theta2", "m21", "gamma1", "mu1", "m22",
        "gamma2", "mu2", "q", "r", "p1", "p2", "cons_1", "cons_2", "cons_3", "pi",
    ];

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

    SimulationParams {
        headers,
        mean,
        cov,
        mu_p,
        sigma_p,
        e_cf,
        var_cf,
        total_iterations: 100,
    }
}

fn initialize_shared_state(
    output_paths: &OutputPaths,
    headers: &[&str],
    worker_count: usize,
) -> SharedState {
    let writers = Arc::new(Mutex::new((
        Writer::from_path(&output_paths.timestamped_csv).expect("Failed to open timestamped CSV"),
        Writer::from_path(&output_paths.global_csv).expect("Failed to open global CSV"),
    )));
    {
        let mut guard = writers.lock().expect("Failed to lock writers");
        guard
            .0
            .write_record(headers)
            .expect("Failed to write timestamped CSV header");
        guard
            .1
            .write_record(headers)
            .expect("Failed to write global CSV header");
    }

    SharedState {
        writers,
        stats: Arc::new(Mutex::new(vec![RunningStats::new(); headers.len()])),
        thread_status: Arc::new(
            (0..worker_count)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        ),
        completed_count: Arc::new(AtomicUsize::new(0)),
        stop_progress: Arc::new(AtomicBool::new(false)),
        skipped_figure_count: Arc::new(AtomicUsize::new(0)),
        csv_error_count: Arc::new(AtomicUsize::new(0)),
        plot_error_count: Arc::new(AtomicUsize::new(0)),
    }
}

fn start_progress_reporter(
    thread_status: Arc<Vec<AtomicUsize>>,
    completed_count: Arc<AtomicUsize>,
    stop_progress: Arc<AtomicBool>,
    worker_count: usize,
    total_iterations: usize,
) -> thread::JoinHandle<()> {
    let progress_lines = worker_count + 1;
    println!("Overall progress: 0/{}", total_iterations);
    for tid in 0..worker_count {
        println!("Thread {:02}: idle", tid);
    }

    thread::spawn(move || {
        let mut stdout = io::stdout();

        loop {
            print!("\x1B[{}A", progress_lines);

            let done = completed_count.load(Ordering::Relaxed);
            print!("\x1B[2K\rOverall progress: {}/{}\n", done, total_iterations);

            for tid in 0..worker_count {
                let current = thread_status[tid].load(Ordering::Relaxed);
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

            if stop_progress.load(Ordering::Relaxed) {
                break;
            }

            thread::sleep(Duration::from_millis(120));
        }
    })
}

fn run_simulation(simulation: &SimulationParams, output_paths: &OutputPaths, shared: &SharedState) {
    (0..simulation.total_iterations)
        .into_par_iter()
        .for_each(|i| {
            run_iteration(
                i,
                rayon::current_thread_index(),
                simulation,
                output_paths,
                shared,
            )
        });
}

fn run_iteration(
    iteration: usize,
    thread_id: Option<usize>,
    simulation: &SimulationParams,
    output_paths: &OutputPaths,
    shared: &SharedState,
) {
    if let Some(tid) = thread_id {
        shared.thread_status[tid].store(iteration + 1, Ordering::Relaxed);
    }

    let (result_data, price_sample, cost_sample) = process_iteration(
        simulation.e_cf,
        simulation.var_cf,
        &simulation.mean,
        &simulation.cov,
    );

    update_running_stats(&shared.stats, &result_data);
    write_result_row(&shared.writers, &result_data, &shared.csv_error_count);
    save_iteration_plot(
        iteration,
        &result_data,
        &price_sample,
        &cost_sample,
        simulation,
        output_paths,
        &shared.skipped_figure_count,
        &shared.plot_error_count,
    );

    shared.completed_count.fetch_add(1, Ordering::Relaxed);
}

fn update_running_stats(stats: &Arc<Mutex<Vec<RunningStats>>>, result_data: &[f64]) {
    if let Ok(mut guard) = stats.lock() {
        for (index, value) in result_data.iter().enumerate() {
            if let Some(column) = guard.get_mut(index) {
                column.update(*value);
            }
        }
    }
}

fn write_result_row(
    writers: &Arc<Mutex<(Writer<std::fs::File>, Writer<std::fs::File>)>>,
    result_data: &[f64],
    csv_error_count: &Arc<AtomicUsize>,
) {
    let row: Vec<String> = result_data.iter().map(|value| value.to_string()).collect();
    match writers.lock() {
        Ok(mut guard) => {
            let write_ok = guard.0.write_record(&row).is_ok()
                && guard.0.flush().is_ok()
                && guard.1.write_record(&row).is_ok()
                && guard.1.flush().is_ok();
            if !write_ok {
                csv_error_count.fetch_add(1, Ordering::Relaxed);
            }
        }
        Err(_) => {
            csv_error_count.fetch_add(1, Ordering::Relaxed);
        }
    }
}

fn save_iteration_plot(
    iteration: usize,
    result_data: &[f64],
    price_sample: &[f64],
    cost_sample: &[f64],
    simulation: &SimulationParams,
    output_paths: &OutputPaths,
    skipped_figure_count: &Arc<AtomicUsize>,
    plot_error_count: &Arc<AtomicUsize>,
) {
    let q = result_data[28];
    let r = result_data[29];
    let p1 = result_data[30];
    let p2 = result_data[31];

    if !(q.is_finite() && r.is_finite() && p1.is_finite() && p2.is_finite()) {
        skipped_figure_count.fetch_add(1, Ordering::Relaxed);
    } else if plotting::save_iteration_figures(
        iteration,
        price_sample,
        cost_sample,
        simulation.mu_p,
        simulation.sigma_p,
        simulation.e_cf,
        simulation.var_cf,
        &output_paths.output_dir,
    )
    .is_err()
    {
        plot_error_count.fetch_add(1, Ordering::Relaxed);
    }
}

fn stop_progress_reporter(stop_progress: &AtomicBool, reporter: thread::JoinHandle<()>) {
    stop_progress.store(true, Ordering::Relaxed);
    let _ = reporter.join();
    println!();
}

fn flush_writers(writers: &Arc<Mutex<(Writer<std::fs::File>, Writer<std::fs::File>)>>) {
    if let Ok(mut guard) = writers.lock() {
        if guard.0.flush().is_err() || guard.1.flush().is_err() {
            println!("Warning: failed to flush CSV writers.");
        }
    }
}

fn collect_final_counts(shared: &SharedState) -> FinalCounts {
    FinalCounts {
        skipped_figures: shared.skipped_figure_count.load(Ordering::Relaxed),
        csv_errors: shared.csv_error_count.load(Ordering::Relaxed),
        plot_errors: shared.plot_error_count.load(Ordering::Relaxed),
    }
}

fn print_run_summary(
    output_paths: &OutputPaths,
    headers: &[&str],
    total_iterations: usize,
    stats: &Arc<Mutex<Vec<RunningStats>>>,
    final_counts: &FinalCounts,
    elapsed: Duration,
) {
    println!("Global results also saved to {}", output_paths.global_csv);
    println!(
        "Figure 1 files saved to {}/figure-1",
        output_paths.output_dir
    );
    println!(
        "Figure 2 files saved to {}/figure-2",
        output_paths.output_dir
    );
    println!(
        "Skipped figure generation for {} iterations due to unsolved q/r/p1/p2",
        final_counts.skipped_figures
    );

    if final_counts.csv_errors > 0 {
        println!(
            "Warning: {} iterations failed to write CSV rows.",
            final_counts.csv_errors
        );
    }

    if final_counts.plot_errors > 0 {
        println!(
            "Warning: {} iterations failed to generate figures.",
            final_counts.plot_errors
        );
    }

    if let Ok(guard) = stats.lock() {
        print_summary_statistics(&guard, headers, total_iterations);
    }

    println!("Results saved to {}", output_paths.timestamped_csv);
    println!("Total time: {:?}", elapsed);
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

    let mean_d = sample_mean(&d_samples);
    let mean_p = sample_mean(&p_samples);
    let mean_v = sample_mean(&v_samples);
    let mean_cf = sample_mean(&cf_samples);
    let var_d = sample_variance(&d_samples, mean_d);
    let var_p = sample_variance(&p_samples, mean_p);
    let var_v = sample_variance(&v_samples, mean_v);
    let var_cf = sample_variance(&cf_samples, mean_cf);

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
