---
title: "FourierLayer<T>"
description: "Represents a single Fourier layer in the FNO."
section: "API Reference"
---

`Layers` · `AiDotNet.PhysicsInformed.NeuralOperators`

Represents a single Fourier layer in the FNO.

## How It Works

For Beginners:
This layer is the heart of the FNO. It performs:

1. FFT: Convert to frequency domain
2. Spectral convolution: Multiply by learned weights (in Fourier space)
3. IFFT: Convert back to physical space
4. Add local convolution (via 1x1 convolution)
5. Apply activation function

Why This Works:

- In Fourier space, convolution becomes multiplication (very efficient!)
- We learn which frequencies are important
- Captures both global (low frequency) and local (high frequency) information

The spectral convolution is key: it's a global operation that couples
all spatial points, allowing the network to capture long-range dependencies.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPointwiseMixing(Tensor<>)` | Tape-tracked 1×1 pointwise mixing across the channel axis. |
| `ApplySeparableFft(Tensor<>,Tensor<>,Boolean)` | Applies an N-D forward or inverse FFT as a sequence of 1-D `Engine.FFT` / `Engine.IFFT` calls, one per spatial axis. |
| `ApplySpectralConvolution2DTape(Tensor<>)` | Tape-tracked spectral convolution for the 2D case using Engine.FFT2D. |
| `ApplySpectralConvolutionNDTape(Tensor<>)` | Tape-tracked spectral convolution for arbitrary spatial rank. |
| `PerLocationMatMul(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32,Int32)` | Per-frequency batched matmul used by the 2D spectral conv. |
| `PerLocationMatMulND(Tensor<>,Tensor<>,Int32,Int32,Int32,Int32[])` | N-D generalization of `Int32)`. |
| `UpdateParameters()` | Legacy scalar-learning-rate parameter update. |

