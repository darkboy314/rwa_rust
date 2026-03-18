mod distribution;
mod ga;
mod game;
mod plotting;
mod utils;

use chrono::Local;
use csv::Writer;
use rayon::prelude::*;
use std::fs;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

fn main() {
    let start = std::time::Instant::now();

    // Create output directory with timestamp
    let timestamp = Local::now().format("%Y-%m-%d %H-%M-%S").to_string();
    let output_dir = format!("./output/{}", timestamp);
    fs::create_dir_all(format!("{}/figure-1", output_dir)).expect("Failed to create figure-1 dir");
    fs::create_dir_all(format!("{}/figure-2", output_dir)).expect("Failed to create figure-2 dir");

    let filename = format!("{}/result.csv", output_dir);

    // Create CSV headers
    let headers = vec![
        "E_D", "E_P", "E_V", "E_DP", "E_PV", "E_cf", "sigma_D", "sigma_P", "sigma_V", "sigma_DP",
        "sigma_PV", "sigma_cf", "m11", "m12", "m21", "m22", "q", "r", "p1", "p2", "fun",
    ];

    // Market parameters
    let e_d = 100.0;
    let e_p = 1400.0;
    let e_v = 100.0;

    let var_d = 3.0;
    let var_p = 1.0;
    let var_v = 2.0;

    let (mu_d, sigma_d) = distribution::lognormal_params_from_mean_var(e_d, var_d);
    let (mu_p, sigma_p) = distribution::lognormal_params_from_mean_var(e_p, var_p);
    let (mu_v, sigma_v) = distribution::lognormal_params_from_mean_var(e_v, var_v);

    let rho_dp = 0.5;
    let rho_pv = 0.3;
    let rho_dv = 0.2;

    let e_cf = 50.0; // For REVB E_cf' = 1.5 * E_cf
    let var_cf = 10.0; //or REVB Var_cf' = 4 * Var_cf (2^2)

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

    let total_iterations = 10000usize;
    println!(
        "Starting RWA simulation with {} iterations...",
        total_iterations
    );

    // Run in single thread for testing
    // process_iteration(e_cf, var_cf, &mean, &cov);

    // Run parallel iterations with in-place per-thread progress display
    let worker_count = rayon::current_num_threads();
    let thread_status = Arc::new(
        (0..worker_count)
            .map(|_| AtomicUsize::new(0))
            .collect::<Vec<_>>(),
    );
    let completed_count = Arc::new(AtomicUsize::new(0));
    let stop_progress = Arc::new(AtomicBool::new(false));

    let progress_lines = worker_count + 1;
    println!("Overall progress: 0/{}", total_iterations);
    for tid in 0..worker_count {
        println!("Thread {:02}: idle", tid);
    }

    let reporter_status = Arc::clone(&thread_status);
    let reporter_completed = Arc::clone(&completed_count);
    let reporter_stop = Arc::clone(&stop_progress);
    let reporter = thread::spawn(move || {
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
    });

    let worker_status = Arc::clone(&thread_status);
    let worker_completed = Arc::clone(&completed_count);

    let iteration_results: Vec<(Vec<f64>, Vec<f64>, Vec<f64>)> = (0..total_iterations)
        .into_par_iter()
        .map(|i| {
            if let Some(tid) = rayon::current_thread_index() {
                worker_status[tid].store(i + 1, Ordering::Relaxed);
            }

            let out = process_iteration(e_cf, var_cf, &mean, &cov);
            worker_completed.fetch_add(1, Ordering::Relaxed);
            out
        })
        .collect();

    stop_progress.store(true, Ordering::Relaxed);
    let _ = reporter.join();
    println!();

    // Separate results and save one figure pair for each iteration
    let mut results: Vec<Vec<f64>> = Vec::with_capacity(10000);
    let mut skipped_figure_count = 0usize;

    for (idx, (result_data, price_sample, cost_sample)) in iteration_results.into_iter().enumerate() {
        results.push(result_data.clone());

        let q = result_data[16];
        let r = result_data[17];
        let p1 = result_data[18];
        let p2 = result_data[19];

        if !(q.is_finite() && r.is_finite() && p1.is_finite() && p2.is_finite()) {
            skipped_figure_count += 1;
            continue;
        }

        if let Err(e) = plotting::save_iteration_figures(
            idx,
            &price_sample,
            &cost_sample,
            mu_p,
            sigma_p,
            e_cf,
            var_cf,
            &output_dir,
        ) {
            println!("Warning: Failed to generate figures for iteration {}: {}", idx, e);
        }
    }

    let global_filename = "./output/result.csv".to_string();

    // Write results to both timestamped and global CSV files
    if let Err(e) = write_results(&results, &headers, &filename) {
        println!("Error writing results to timestamped CSV: {}", e);
        return;
    }

    if let Err(e) = write_results(&results, &headers, &global_filename) {
        println!("Error writing results to global CSV: {}", e);
    } else {
        println!("Global results also saved to {}", global_filename);
    }

    println!("Figure 1 files saved to {}/figure-1", output_dir);
    println!("Figure 2 files saved to {}/figure-2", output_dir);
    println!(
        "Skipped figure generation for {} iterations due to unsolved q/r/p1/p2",
        skipped_figure_count
    );

    // Print summary statistics
    print_summary_statistics(&results, &headers);

    println!("Results saved to {}", filename);
    println!("Total time: {:?}", start.elapsed());
}

fn print_summary_statistics(results: &[Vec<f64>], headers: &[&str]) {
    if results.is_empty() {
        return;
    }

    println!("\n=== Summary Statistics ===");
    println!("Total iterations: {}", results.len());

    for (i, &header) in headers.iter().enumerate() {
        if i >= results[0].len() {
            break;
        }

        let values: Vec<f64> = results.iter().map(|row| row[i]).collect();
        let mean = utils::mean(&values);
        let std = utils::std_dev(&values);
        let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

        println!(
            "{}: mean={:.6}, std={:.6}, min={:.6}, max={:.6}",
            header, mean, std, min_val, max_val
        );
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
    let cf = distribution::operation_cost_gamma(100, e_cf, var_cf, Some(0));
    let dpv = distribution::sample_multivariate_lognormal(100, mean, cov, Some(0));

    // Calculate statistics
    let mean_d = utils::mean(&dpv.0);
    let mean_p = utils::mean(&dpv.1);
    let mean_v = utils::mean(&dpv.2);
    let mean_dp = utils::mean(&utils::elementwise_mul(&dpv.0, &dpv.1));
    let mean_pv = utils::mean(&utils::elementwise_mul(&dpv.1, &dpv.2));

    let s_d = utils::std_dev(&dpv.0);
    let s_p = utils::std_dev(&dpv.1);
    let s_v = utils::std_dev(&dpv.2);
    let s_dp = utils::std_dev(&utils::elementwise_mul(&dpv.0, &dpv.1));
    let s_pv = utils::std_dev(&utils::elementwise_mul(&dpv.1, &dpv.2));

    let mean_cf = utils::mean(&cf);
    let s_cf = utils::std_dev(&cf);

    // Run game simulation
    let (m11, m12, m21, m22, q, r, p1, p2, fun) = game::start_game(
        100, mean_d, mean_p, mean_v, mean_dp, mean_pv, mean_cf, s_cf, s_d, s_p, s_v, s_dp, s_pv,
    );

    // Collect results
    let result_data = vec![
        mean_d, mean_p, mean_v, mean_dp, mean_pv, mean_cf, s_d, s_p, s_v, s_dp, s_pv, s_cf, m11,
        m12, m21, m22, q, r, p1, p2, fun,
    ];

    // Return result data with per-iteration samples for plotting
    (result_data, dpv.1.clone(), cf)
}

fn write_results(
    results: &[Vec<f64>],
    headers: &[&str],
    filename: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    use std::path::Path;

    let file_exists = Path::new(filename).exists();
    let mut wtr = Writer::from_path(filename)?;

    // Write headers only if file doesn't exist
    if !file_exists {
        wtr.write_record(headers)?;
    }

    // Write data
    for row in results {
        let row_str: Vec<String> = row.iter().map(|v| format!("{}", v)).collect();
        wtr.write_record(&row_str)?;
    }

    wtr.flush()?;
    Ok(())
}

