#[inline]
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().sum::<f64>() / values.len() as f64
}

#[inline]
pub fn variance(values: &[f64], mean: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / values.len() as f64
}