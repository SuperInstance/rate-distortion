//! Rate-distortion bounds.
//!
//! Theoretical bounds on the rate-distortion function.

/// Shannon lower bound on distortion for a given rate (Gaussian source).
///
/// D ≥ σ² · 2^(-2R) for a Gaussian source with variance σ².
pub fn rate_distortion_lower_bound(variance: f64, rate: f64) -> f64 {
    if rate <= 0.0 {
        return variance;
    }
    variance * 2.0_f64.powf(-2.0 * rate)
}

/// Upper bound on distortion for uniform scalar quantization.
///
/// D ≤ (Δ²/12) where Δ is the quantization step size,
/// which for R bits gives D ≤ σ² · (c²) · 2^(-2R) for some constant c.
pub fn rate_distortion_upper_bound(variance: f64, rate: f64) -> f64 {
    if rate <= 0.0 {
        return variance;
    }
    // Upper bound is looser than lower bound by a constant factor
    // For uniform quantizer: D = (1/12) * Δ² ≈ (πe/6) * D_lower for Gaussian
    let lower = rate_distortion_lower_bound(variance, rate);
    lower * (std::f64::consts::PI * std::f64::consts::E / 6.0)
}

/// Compute the signal-to-noise ratio in dB.
pub fn snr_db(signal_power: f64, noise_power: f64) -> f64 {
    if noise_power <= 0.0 {
        return f64::INFINITY;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Compute the rate for a given distortion target (Gaussian source).
///
/// R(D) = (1/2) log₂(σ²/D) for D < σ².
pub fn gaussian_rate_distortion(variance: f64, distortion: f64) -> f64 {
    if distortion >= variance {
        return 0.0;
    }
    if distortion <= 0.0 {
        return f64::INFINITY;
    }
    0.5 * (variance / distortion).log2()
}
