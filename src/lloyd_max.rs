//! Lloyd-Max algorithm.
//!
//! Optimal quantizer design using the Lloyd-Max iterative algorithm.

use crate::quantizer::ScalarQuantizer;

/// Run the Lloyd-Max algorithm to design an optimal quantizer.
///
/// Given training samples and a target number of levels, iteratively
/// refines boundaries and reconstruction levels to minimize MSE.
pub fn lloyd_max(samples: &[f64], num_levels: usize, iterations: usize) -> ScalarQuantizer {
    assert!(num_levels >= 2, "need at least 2 levels");
    assert!(!samples.is_empty(), "need training samples");

    let lo = samples.iter().cloned().fold(f64::INFINITY, f64::min);
    let hi = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // Initialize uniformly
    let step = (hi - lo) / num_levels as f64;
    let mut boundaries: Vec<f64> = (0..=num_levels).map(|i| lo + i as f64 * step).collect();
    let mut levels: Vec<f64> = (0..num_levels).map(|i| lo + (i as f64 + 0.5) * step).collect();

    for _ in 0..iterations {
        // Assign samples to levels
        let mut sums = vec![0.0f64; num_levels];
        let mut counts = vec![0usize; num_levels];

        for &x in samples {
            let mut idx = 0;
            for i in 0..num_levels {
                if x < boundaries[i + 1] {
                    idx = i;
                    break;
                }
                idx = i;
            }
            sums[idx] += x;
            counts[idx] += 1;
        }

        // Update levels (centroids)
        for i in 0..num_levels {
            if counts[i] > 0 {
                levels[i] = sums[i] / counts[i] as f64;
            }
        }

        // Update boundaries (midpoints between adjacent levels)
        for i in 1..num_levels {
            boundaries[i] = (levels[i - 1] + levels[i]) / 2.0;
        }
    }

    ScalarQuantizer::new(boundaries, levels)
}
