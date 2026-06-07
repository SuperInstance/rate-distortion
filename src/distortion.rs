//! Distortion measures.
//!
//! Common distortion measures including Hamming distance and squared error.

/// Trait for custom distortion measures.
pub trait DistortionMeasure {
    /// Compute the distortion between two sequences.
    fn measure(&self, x: &[f64], y: &[f64]) -> f64;
}

/// Compute normalized Hamming distance between two integer sequences.
pub fn hamming_distance(x: &[i32], y: &[i32]) -> f64 {
    assert_eq!(x.len(), y.len(), "sequences must have equal length");
    let mismatches = x.iter().zip(y.iter())
        .filter(|(a, b)| a != b)
        .count();
    mismatches as f64 / x.len() as f64
}

/// Compute average squared error between two sequences.
pub fn squared_error(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "sequences must have equal length");
    let sum: f64 = x.iter().zip(y.iter())
        .map(|(a, b)| (a - b) * (a - b))
        .sum();
    sum / x.len() as f64
}

/// Compute mean squared error.
pub fn mean_squared_error(x: &[f64], y: &[f64]) -> f64 {
    squared_error(x, y)
}

/// Compute absolute error.
pub fn absolute_error(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "sequences must have equal length");
    let sum: f64 = x.iter().zip(y.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    sum / x.len() as f64
}
