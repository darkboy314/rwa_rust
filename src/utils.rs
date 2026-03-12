/// Calculate mean of a vector
pub fn mean(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// Calculate standard deviation (sample std dev)
pub fn std_dev(data: &[f64]) -> f64 {
    if data.len() < 2 {
        return 0.0;
    }

    let m = mean(data);
    let variance = data.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

    variance.sqrt()
}

/// Element-wise multiplication
pub fn elementwise_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

/// Gamma function approximation using Lanczos approximation
pub fn gamma(z: f64) -> f64 {
    use std::f64::consts::PI;

    if z < 0.5 {
        PI / ((PI * z).sin() * gamma(1.0 - z))
    } else {
        let g = 7.0;
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

        let z = z - 1.0;
        let mut x = p[0];
        for (i, &coeff) in p.iter().enumerate().skip(1) {
            x += coeff / (z + i as f64);
        }
        let t = z + g + 0.5;
        (2.0 * PI).sqrt() * t.powf(z + 0.5) * (-t).exp() * x
    }
}
