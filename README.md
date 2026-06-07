# rate-distortion

Rate-distortion theory algorithms in pure Rust with no external dependencies.

## Features

- **Distortion measures**: Hamming distance, squared error, absolute error
- **Blahut-Arimoto algorithm**: Iterative R(D) function computation
- **Scalar quantizer**: Uniform and custom quantization
- **Lloyd-Max algorithm**: Optimal quantizer design via iterative refinement
- **Theoretical bounds**: Shannon lower bound, rate-distortion bounds for Gaussian sources

## Usage

```rust
use rate_distortion::*;

// Distortion measures
let d = hamming_distance(&[1, 2, 3], &[1, 3, 3]);
let e = squared_error(&[0.0, 2.0], &[2.0, 0.0]);

// Lloyd-Max quantizer
let samples: Vec<f64> = (0..1000).map(|i| (i as f64 / 1000.0) * 2.0 - 1.0).collect();
let q = lloyd_max(&samples, 4, 50);
let idx = q.quantize(0.3);
let reconstructed = q.dequantize(idx);

// Blahut-Arimoto
let px = vec![0.5, 0.5];
let channel = vec![vec![0.9, 0.1], vec![0.1, 0.9]];
let (rate, distortion) = blahut_arimoto(&px, &channel, 100);
```

## License

MIT
