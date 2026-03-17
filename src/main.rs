mod distribution;
mod ga;
mod game;
mod utils;

use chrono::Local;
use csv::Writer;
use plotters::prelude::*;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::fs;
use std::io::{self, Write};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

fn lognormal_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        let ln_x = x.ln();
        let coeff = 1.0 / (x * sigma * (2.0 * PI).sqrt());
        let exponent = -((ln_x - mu).powi(2)) / (2.0 * sigma.powi(2));
        coeff * exponent.exp()
    }
}

fn gamma_pdf(x: f64, k: f64, theta: f64) -> f64 {
    if x <= 0.0 {
        0.0
    } else {
        let coeff = x.powf(k - 1.0) * (-x / theta).exp() / (theta.powf(k) * utils::gamma(k));
        coeff
    }
}

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

    let iteration_results: Vec<(Vec<f64>, Vec<f64>)> = (0..total_iterations)
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

    // Separate results and samples
    let mut results: Vec<Vec<f64>> = Vec::with_capacity(10000);
    let mut price_samples: Vec<f64> = Vec::new();
    let mut cost_samples: Vec<f64> = Vec::new();

    for (result_data, price_sample) in iteration_results {
        results.push(result_data.clone());
        price_samples.extend(price_sample);

        // Generate cost samples for figure 2 (using first iteration as representative)
        if cost_samples.is_empty() {
            cost_samples = distribution::operation_cost_gamma(100, e_cf, var_cf, Some(0));
        }

        // Print iteration results like Python version
        // let q = result_data[16];
        // let r = result_data[17];
        // let p1 = result_data[18];
        // let p2 = result_data[19];
        // let fun = result_data[20];
        // println!("q = {:.6}, r = {:.6}, p1 = {:.6}, p2 = {:.6}, pi = {:.6}", q, r, p1, p2, fun);
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

    // Generate figures
    if let Err(e) = generate_figure_1(&price_samples, mu_p, sigma_p, &output_dir) {
        println!("Warning: Failed to generate Figure 1: {}", e);
    } else {
        println!(
            "Figure 1 saved to {}/figure-1/price_distribution.png",
            output_dir
        );
    }

    if let Err(e) = generate_figure_2(&cost_samples, e_cf, var_cf, &output_dir) {
        println!("Warning: Failed to generate Figure 2: {}", e);
    } else {
        println!(
            "Figure 2 saved to {}/figure-2/cost_distribution.png",
            output_dir
        );
    }

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
) -> (Vec<f64>, Vec<f64>) {
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

    // Return both result data and price samples for plotting
    (result_data, dpv.1.clone())
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

fn generate_figure_1(
    price_samples: &[f64],
    mu_p: f64,
    sigma_p: f64,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let figure_path = format!("{}/figure-1/price_distribution.png", output_dir);

    let root = BitMapBackend::new(&figure_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = price_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = price_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Figure 1 - Distribution of Price Samples and PDF",
            ("sans-serif", 20),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_val..max_val, 0.0..100.0)?;

    chart.configure_mesh().draw()?;

    // Create histogram data manually
    let bin_count = 20;
    let bin_width = (max_val - min_val) / bin_count as f64;
    let mut bins = vec![0u32; bin_count];

    for &sample in price_samples {
        let bin_idx = ((sample - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bin_count - 1);
        bins[bin_idx] += 1;
    }

    // Draw histogram as bars
    for (i, &count) in bins.iter().enumerate() {
        let x0 = min_val + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        let y = count as f64;

        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, y)],
            BLUE.mix(0.5).filled(),
        )))?;
    }

    // Draw theoretical log-normal PDF
    let pdf_points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let x = min_val + (max_val - min_val) * i as f64 / 99.0;
            let pdf_val = lognormal_pdf(x, mu_p, sigma_p) * 100.0; // Scale for visibility
            (x, pdf_val)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(pdf_points, &RED))?
        .label(format!("Log-normal PDF (μ={:.2}, σ={:.3})", mu_p, sigma_p))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

    chart.configure_series_labels().draw()?;

    root.present()?;
    Ok(())
}

fn generate_figure_2(
    cost_samples: &[f64],
    e_cf: f64,
    var_cf: f64,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let figure_path = format!("{}/figure-2/cost_distribution.png", output_dir);

    let root = BitMapBackend::new(&figure_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = cost_samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = cost_samples
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    let mut chart = ChartBuilder::on(&root)
        .caption(
            "Figure 2 - Distribution of Operating Cost Samples and PDF",
            ("sans-serif", 20),
        )
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_val..max_val, 0.0..50.0)?;

    chart.configure_mesh().draw()?;

    // Create histogram data manually
    let bin_count = 20;
    let bin_width = (max_val - min_val) / bin_count as f64;
    let mut bins = vec![0u32; bin_count];

    for &sample in cost_samples {
        let bin_idx = ((sample - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bin_count - 1);
        bins[bin_idx] += 1;
    }

    // Draw histogram as bars
    for (i, &count) in bins.iter().enumerate() {
        let x0 = min_val + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        let y = count as f64;

        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, y)],
            RED.mix(0.5).filled(),
        )))?;
    }

    // Draw theoretical gamma PDF
    let theta = var_cf / e_cf;
    let k = e_cf / theta;
    let pdf_points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let x = min_val + (max_val - min_val) * i as f64 / 99.0;
            let pdf_val = gamma_pdf(x, k, theta) * 1000.0; // Scale for visibility
            (x, pdf_val)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(pdf_points, &BLUE))?
        .label(format!("Gamma PDF (k={:.2}, θ={:.3})", k, theta))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

    chart.configure_series_labels().draw()?;

    root.present()?;
    Ok(())
}
