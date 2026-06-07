//! Scalar quantizer design.
//!
//! Uniform and custom scalar quantization.

/// A scalar quantizer with defined boundaries and reconstruction levels.
#[derive(Debug, Clone)]
pub struct ScalarQuantizer {
    /// Decision boundaries (num_levels + 1, including -inf and +inf ends conceptually)
    boundaries: Vec<f64>,
    /// Reconstruction levels
    levels: Vec<f64>,
}

impl ScalarQuantizer {
    /// Create a uniform quantizer with `n` levels over [lo, hi].
    pub fn uniform(n: usize, lo: f64, hi: f64) -> Self {
        let step = (hi - lo) / n as f64;
        let boundaries: Vec<f64> = (0..=n).map(|i| lo + i as f64 * step).collect();
        let levels: Vec<f64> = (0..n).map(|i| lo + (i as f64 + 0.5) * step).collect();
        Self { boundaries, levels }
    }

    /// Create a quantizer from explicit boundaries and levels.
    pub fn new(boundaries: Vec<f64>, levels: Vec<f64>) -> Self {
        assert_eq!(boundaries.len(), levels.len() + 1);
        Self { boundaries, levels }
    }

    /// Quantize a value, returning the level index.
    pub fn quantize(&self, x: f64) -> usize {
        for i in 0..self.levels.len() {
            if x < self.boundaries[i + 1] {
                return i;
            }
        }
        self.levels.len() - 1
    }

    /// Dequantize a level index back to the reconstruction value.
    pub fn dequantize(&self, idx: usize) -> f64 {
        self.levels[idx]
    }

    /// Get the number of quantization levels.
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get the reconstruction levels.
    pub fn levels(&self) -> &[f64] {
        &self.levels
    }

    /// Get the decision boundaries.
    pub fn boundaries(&self) -> &[f64] {
        &self.boundaries
    }

    /// Compute the step size (for uniform quantizers).
    pub fn step_size(&self) -> f64 {
        if self.boundaries.len() < 2 {
            return 0.0;
        }
        self.boundaries[1] - self.boundaries[0]
    }
}

/// Quantize a slice of values, returning indices.
pub fn quantize(data: &[f64], q: &ScalarQuantizer) -> Vec<usize> {
    data.iter().map(|&x| q.quantize(x)).collect()
}

/// Dequantize a slice of indices.
pub fn dequantize(indices: &[usize], q: &ScalarQuantizer) -> Vec<f64> {
    indices.iter().map(|&i| q.dequantize(i)).collect()
}
