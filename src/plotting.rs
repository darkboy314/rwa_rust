use plotters::prelude::*;
use std::error::Error;
use std::f64::consts::PI;

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
    if x <= 0.0 || k <= 0.0 || theta <= 0.0 {
        0.0
    } else {
        // Evaluate in log-space for numerical stability when k is large.
        let ln_pdf = (k - 1.0) * x.ln() - x / theta - k * theta.ln() - ln_gamma_lanczos(k);
        ln_pdf.exp()
    }
}

fn ln_gamma_lanczos(z: f64) -> f64 {
    // Lanczos approximation in log domain.
    let p = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let g = 7.0;

    if z < 0.5 {
        // Reflection formula
        let pi = std::f64::consts::PI;
        (pi / ((pi * z).sin())).ln() - ln_gamma_lanczos(1.0 - z)
    } else {
        let zz = z - 1.0;
        let mut x = p[0];
        for (i, &coeff) in p.iter().enumerate().skip(1) {
            x += coeff / (zz + i as f64);
        }
        let t = zz + g + 0.5;
        0.5 * (2.0 * std::f64::consts::PI).ln() + (zz + 0.5) * t.ln() - t + x.ln()
    }
}

pub fn save_iteration_figures(
    iteration: usize,
    price_samples: &[f64],
    cost_samples: &[f64],
    mu_p: f64,
    sigma_p: f64,
    e_cf: f64,
    var_cf: f64,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    save_price_figure(iteration, price_samples, mu_p, sigma_p, output_dir)?;
    save_cost_figure(iteration, cost_samples, e_cf, var_cf, output_dir)?;
    Ok(())
}

fn save_price_figure(
    iteration: usize,
    price_samples: &[f64],
    mu_p: f64,
    sigma_p: f64,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    let figure_path = format!("{}/figure-1/{}.png", output_dir, iteration);
    let root = BitMapBackend::new(&figure_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = price_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = price_samples
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let (min_x, max_x) = if min_val < max_val {
        (min_val, max_val)
    } else {
        (min_val - 1.0, max_val + 1.0)
    };

    let mut chart = ChartBuilder::on(&root)
        .caption("Figure 1 - Distribution of Pm and sample", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, 0.0..(price_samples.len() as f64).max(1.0))?;

    chart.configure_mesh().draw()?;

    let bin_count = 10usize;
    let bin_width = (max_x - min_x) / bin_count as f64;
    let mut bins = vec![0u32; bin_count];

    for &sample in price_samples {
        let bin_idx = ((sample - min_x) / bin_width).floor() as usize;
        let clamped = bin_idx.min(bin_count - 1);
        bins[clamped] += 1;
    }

    for (i, &count) in bins.iter().enumerate() {
        let x0 = min_x + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f64)],
            BLUE.mix(0.5).filled(),
        )))?;
    }

    let pdf_points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let x = min_x + (max_x - min_x) * i as f64 / 99.0;
            let pdf_val = lognormal_pdf(x, mu_p, sigma_p) * price_samples.len() as f64;
            (x, pdf_val)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(pdf_points, &RED))?
        .label(format!("Log-normal PDF (mu={:.3}, sigma={:.3})", mu_p, sigma_p))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], RED));

    chart.configure_series_labels().draw()?;
    root.present()?;
    Ok(())
}

fn save_cost_figure(
    iteration: usize,
    cost_samples: &[f64],
    e_cf: f64,
    var_cf: f64,
    output_dir: &str,
) -> Result<(), Box<dyn Error>> {
    let figure_path = format!("{}/figure-2/{}.png", output_dir, iteration);
    let root = BitMapBackend::new(&figure_path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let min_val = cost_samples.iter().copied().fold(f64::INFINITY, f64::min);
    let max_val = cost_samples
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    let (min_x, max_x) = if min_val < max_val {
        (min_val, max_val)
    } else {
        (min_val - 1.0, max_val + 1.0)
    };

    let mut chart = ChartBuilder::on(&root)
        .caption("Figure 2 - Distribution of cf and sample", ("sans-serif", 20))
        .margin(5)
        .x_label_area_size(30)
        .y_label_area_size(30)
        .build_cartesian_2d(min_x..max_x, 0.0..(cost_samples.len() as f64).max(1.0))?;

    chart.configure_mesh().draw()?;

    let bin_count = 10usize;
    let bin_width = (max_x - min_x) / bin_count as f64;
    let mut bins = vec![0u32; bin_count];

    for &sample in cost_samples {
        let bin_idx = ((sample - min_x) / bin_width).floor() as usize;
        let clamped = bin_idx.min(bin_count - 1);
        bins[clamped] += 1;
    }

    for (i, &count) in bins.iter().enumerate() {
        let x0 = min_x + i as f64 * bin_width;
        let x1 = x0 + bin_width;
        chart.draw_series(std::iter::once(Rectangle::new(
            [(x0, 0.0), (x1, count as f64)],
            RED.mix(0.5).filled(),
        )))?;
    }

    let theta = var_cf / e_cf;
    let k = e_cf / theta;
    let pdf_points: Vec<(f64, f64)> = (0..100)
        .map(|i| {
            let x = min_x + (max_x - min_x) * i as f64 / 99.0;
            // Convert density to expected bin count to match histogram count axis.
            let pdf_val = gamma_pdf(x, k, theta) * cost_samples.len() as f64 * bin_width;
            (x, pdf_val)
        })
        .collect();

    chart
        .draw_series(LineSeries::new(pdf_points, &BLUE))?
        .label(format!("Gamma PDF (k={:.3}, theta={:.3})", k, theta))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLUE));

    chart.configure_series_labels().draw()?;
    root.present()?;
    Ok(())
}
