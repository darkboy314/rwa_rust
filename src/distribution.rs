use rand::SeedableRng;
use rand_distr::Distribution;
use rand_distr::{Gamma, Normal};

/// Calculate lognormal parameters from mean and variance
/// Given X ~ LogNormal(μ, σ²), where:
/// mean = exp(μ + σ²/2)
/// var = (exp(σ²) - 1) * exp(2μ + σ²)
///
/// We solve for μ and σ given mean and var:
/// σ² = ln(1 + var/mean²)
/// μ = ln(mean) - σ²/2
pub fn lognormal_params_from_mean_var(mean: f64, var: f64) -> (f64, f64) {
    let sigma2 = (1.0 + var / (mean * mean)).ln();
    let mu = mean.ln() - sigma2 / 2.0;
    let sigma = sigma2.sqrt();
    (mu, sigma)
}

/// Sample from Gamma distribution
pub fn operation_cost_gamma(num: usize, e: f64, var: f64, seed: Option<u64>) -> Vec<f64> {
    let mut rng = if let Some(s) = seed {
        rand_chacha::ChaCha8Rng::seed_from_u64(s)
    } else {
        rand_chacha::ChaCha8Rng::seed_from_u64(rand::random())
    };

    let theta = var / e;
    let k = e / theta;

    let dist = Gamma::new(k, theta).unwrap();
    (0..num).map(|_| dist.sample(&mut rng)).collect()
}

/// Sample from bivariate/multivariate lognormal distribution
/// Returns (Y1, Y2, Y3) samples
pub fn sample_multivariate_lognormal(
    n_samples: usize,
    mean_log: &[f64],
    cov_log: &[Vec<f64>],
    seed: Option<u64>,
) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut rng = if let Some(s) = seed {
        rand_chacha::ChaCha8Rng::seed_from_u64(s)
    } else {
        rand_chacha::ChaCha8Rng::seed_from_u64(rand::random())
    };

    // Cholesky decomposition of covariance matrix
    let l = cholesky_decomposition(cov_log);

    let mut y1 = Vec::new();
    let mut y2 = Vec::new();
    let mut y3 = Vec::new();

    let normal = Normal::new(0.0, 1.0).unwrap();

    for _ in 0..n_samples {
        // Generate standard normal samples
        let z = vec![
            normal.sample(&mut rng),
            normal.sample(&mut rng),
            normal.sample(&mut rng),
        ];

        // Apply Cholesky: x = mean + L @ z
        let mut x = vec![0.0; 3];
        for i in 0..3 {
            x[i] = mean_log[i];
            for j in 0..=i {
                x[i] += l[i][j] * z[j];
            }
        }

        // Exponentiate to get lognormal samples
        y1.push(x[0].exp());
        y2.push(x[1].exp());
        y3.push(x[2].exp());
    }

    (y1, y2, y3)
}

/// Cholesky decomposition for 3x3 matrix
fn cholesky_decomposition(cov: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let mut l = vec![vec![0.0; 3]; 3];

    for i in 0..3 {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }

            if i == j {
                l[i][j] = (cov[i][i] - sum).sqrt();
            } else {
                if l[j][j].abs() > 1e-10 {
                    l[i][j] = (cov[i][j] - sum) / l[j][j];
                }
            }
        }
    }

    l
}

// Re-export for rand_chacha
use rand_chacha;
