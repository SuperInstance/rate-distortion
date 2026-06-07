//! # rate-distortion
//!
//! Rate-distortion theory algorithms in pure Rust.
//!
//! Includes distortion measures (Hamming, squared error), Blahut-Arimoto
//! algorithm for R(D), and Lloyd-Max quantizer design.

pub mod distortion;
pub mod blahut_arimoto;
pub mod quantizer;
pub mod lloyd_max;
pub mod bounds;

pub use distortion::{hamming_distance, squared_error, DistortionMeasure};
pub use blahut_arimoto::blahut_arimoto;
pub use quantizer::{ScalarQuantizer, quantize, dequantize};
pub use lloyd_max::lloyd_max;
pub use bounds::{rate_distortion_lower_bound, rate_distortion_upper_bound};

#[cfg(test)]
mod tests {
    use super::*;

    // === Distortion tests ===
    #[test]
    fn test_hamming_identical() {
        assert_eq!(hamming_distance(&[1, 2, 3], &[1, 2, 3]), 0.0);
    }

    #[test]
    fn test_hamming_different() {
        let d = hamming_distance(&[1, 2, 3], &[1, 3, 3]);
        assert!((d - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_error_identical() {
        assert_eq!(squared_error(&[1.0, 2.0], &[1.0, 2.0]), 0.0);
    }

    #[test]
    fn test_squared_error_basic() {
        let d = squared_error(&[0.0], &[3.0]);
        assert!((d - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_squared_error_average() {
        let d = squared_error(&[0.0, 2.0], &[2.0, 0.0]);
        // (4 + 4) / 2 = 4.0
        assert!((d - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_distortion_measure_trait() {
        struct Hamming;
        impl DistortionMeasure for Hamming {
            fn measure(&self, x: &[f64], y: &[f64]) -> f64 {
                x.iter().zip(y.iter())
                    .map(|(a, b)| if (a - b).abs() > 0.5 { 1.0 } else { 0.0 })
                    .sum::<f64>() / x.len() as f64
            }
        }
        let h = Hamming;
        assert!((h.measure(&[0.0, 1.0], &[0.0, 1.0]) - 0.0).abs() < 1e-10);
        assert!((h.measure(&[0.0, 1.0], &[1.0, 0.0]) - 1.0).abs() < 1e-10);
    }

    // === Blahut-Arimoto tests ===
    #[test]
    fn test_blahut_arimoto_bsc() {
        // Binary symmetric channel with crossover 0.1
        let px = vec![0.5, 0.5];
        let channel = vec![
            vec![0.9, 0.1],
            vec![0.1, 0.9],
        ];
        let (rate, distortion) = blahut_arimoto(&px, &channel, 100);
        assert!(rate > 0.0);
        assert!(distortion >= 0.0);
    }

    #[test]
    fn test_blahut_arimoto_convergence() {
        let px = vec![0.5, 0.5];
        let channel = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
        ];
        let (rate, distortion) = blahut_arimoto(&px, &channel, 50);
        // Perfect channel: rate should be the entropy, distortion ~ 0
        assert!((distortion - 0.0).abs() < 0.01);
    }

    #[test]
    fn test_blahut_arimoto_uniform() {
        let n = 4;
        let px = vec![1.0 / n as f64; n];
        let channel: Vec<Vec<f64>> = (0..n).map(|i| {
            (0..n).map(|j| if i == j { 0.8 } else { 0.2 / (n - 1) as f64 }).collect()
        }).collect();
        let (rate, _) = blahut_arimoto(&px, &channel, 100);
        assert!(rate > 0.0 && rate.is_finite());
    }

    // === Quantizer tests ===
    #[test]
    fn test_quantizer_basic() {
        let q = ScalarQuantizer::uniform(4, -1.0, 1.0);
        let idx = q.quantize(0.3);
        assert!(idx < 4);
    }

    #[test]
    fn test_quantizer_boundaries() {
        let q = ScalarQuantizer::uniform(2, 0.0, 1.0);
        assert_eq!(q.quantize(0.0), 0);
        assert_eq!(q.quantize(0.49), 0);
        assert_eq!(q.quantize(0.51), 1);
    }

    #[test]
    fn test_quantizer_roundtrip_error() {
        let q = ScalarQuantizer::uniform(8, -2.0, 2.0);
        let x = 1.5f64;
        let idx = q.quantize(x);
        let y = q.dequantize(idx);
        assert!((x - y).abs() < 0.5); // Error < step size
    }

    // === Lloyd-Max tests ===
    #[test]
    fn test_lloyd_max_convergence() {
        // Uniform source, 4 levels
        let samples: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * 2.0 - 1.0).collect();
        let q = lloyd_max(&samples, 4, 50);
        assert_eq!(q.num_levels(), 4);
    }

    #[test]
    fn test_lloyd_max_gaussian_like() {
        // Approximate Gaussian via simple generation
        let mut samples = Vec::new();
        for i in -500..500i32 {
            let x = i as f64 / 100.0;
            // Crude Gaussian-ish samples
            samples.push(x * x.cos());
        }
        let q = lloyd_max(&samples, 3, 100);
        assert_eq!(q.num_levels(), 3);
    }

    // === Bounds tests ===
    #[test]
    fn test_lower_bound_zero_rate() {
        // At rate = 0, distortion should be variance
        let variance = 1.0;
        let lb = rate_distortion_lower_bound(variance, 0.0);
        assert!((lb - variance).abs() < 1e-10);
    }

    #[test]
    fn test_upper_bound_high_rate() {
        let variance = 1.0;
        let ub = rate_distortion_upper_bound(variance, 10.0);
        assert!(ub > 0.0);
        assert!(ub < variance);
    }

    #[test]
    fn test_bounds_ordering() {
        let variance = 2.0;
        let rate = 2.0;
        let lb = rate_distortion_lower_bound(variance, rate);
        let ub = rate_distortion_upper_bound(variance, rate);
        assert!(lb <= ub + 1e-10);
    }
}
