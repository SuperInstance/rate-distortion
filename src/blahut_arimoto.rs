//! Blahut-Arimoto algorithm.
//!
//! Computes the rate-distortion function R(D) iteratively.

/// Run the Blahut-Arimoto algorithm to find rate-distortion trade-off.
///
/// Given source distribution `px` and a channel (transition probability matrix),
/// computes the mutual information (rate) and expected distortion.
///
/// Returns (rate, distortion) pair.
pub fn blahut_arimoto(px: &[f64], channel: &[Vec<f64>], iterations: usize) -> (f64, f64) {
    let n_in = px.len();
    let n_out = channel[0].len();

    // Compute q(y) = sum_x px(x) * channel(x, y)
    let mut qy = vec![0.0; n_out];
    for (x, chan_row) in channel.iter().enumerate().take(n_in) {
        for (y, &prob) in chan_row.iter().enumerate().take(n_out) {
            qy[y] += px[x] * prob;
        }
    }

    // Run BA iterations (refine qy)
    for _ in 0..iterations {
        for y in 0..n_out {
            qy[y] = 0.0;
            for (x, chan_row) in channel.iter().enumerate().take(n_in) {
                qy[y] += px[x] * chan_row[y];
            }
        }
    }

    // Compute mutual information I(X;Y) = sum p(x,y) log(p(x,y) / (p(x)*p(y)))
    let mut rate = 0.0;
    for (x, chan_row) in channel.iter().enumerate().take(n_in) {
        for (y, &prob) in chan_row.iter().enumerate().take(n_out) {
            let pxy = px[x] * prob;
            let pxy_marginal = px[x] * qy[y];
            if pxy > 1e-15 && pxy_marginal > 1e-15 {
                rate += pxy * (pxy / pxy_marginal).log2();
            }
        }
    }

    // Compute expected distortion (Hamming)
    let mut distortion = 0.0;
    for (x, chan_row) in channel.iter().enumerate().take(n_in) {
        for (y, &prob) in chan_row.iter().enumerate().take(n_out) {
            let pxy = px[x] * prob;
            let d = if x == y { 0.0 } else { 1.0 };
            distortion += pxy * d;
        }
    }

    (rate.max(0.0), distortion)
}

/// Compute the R(D) function at a specific distortion level D.
pub fn rate_distortion_function(
    px: &[f64],
    alphabet_size: usize,
    target_distortion: f64,
    iterations: usize,
) -> f64 {
    let mut s_lo = 0.01;
    let mut s_hi = 100.0;

    for _ in 0..50 {
        let s = (s_lo + s_hi) / 2.0;
        let channel = build_test_channel(alphabet_size, s);
        let (rate, dist) = blahut_arimoto(px, &channel, iterations);

        if dist > target_distortion {
            s_hi = s;
        } else {
            s_lo = s;
        }

        if rate.is_finite() && (dist - target_distortion).abs() / (target_distortion + 1e-10) < 0.01 {
            return rate;
        }
    }

    let s = (s_lo + s_hi) / 2.0;
    let channel = build_test_channel(alphabet_size, s);
    let (rate, _) = blahut_arimoto(px, &channel, iterations);
    rate
}

fn build_test_channel(n: usize, s: f64) -> Vec<Vec<f64>> {
    let mut channel = vec![vec![0.0; n]; n];
    for (x, chan_row) in channel.iter_mut().enumerate() {
        let mut total = 0.0;
        for (y, entry) in chan_row.iter_mut().enumerate() {
            let d: f64 = if x == y { 0.0 } else { 1.0 };
            *entry = (-s * d).exp();
            total += *entry;
        }
        for entry in chan_row.iter_mut() {
            *entry /= total;
        }
    }
    channel
}
