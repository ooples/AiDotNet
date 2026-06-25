---
title: "Conv1DTransposeLayer<T>"
description: "1D transposed convolution (\"deconvolution\") for sequence / waveform data — the learnable temporal-upsampling primitive used by HiFi-GAN (Kong et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

1D transposed convolution ("deconvolution") for sequence / waveform data —
the learnable temporal-upsampling primitive used by HiFi-GAN (Kong et al.
2020) and the GAN-vocoder family. Operates on rank-3 input
`[B, C_in, T]` and produces rank-3 output `[B, C_out, T_out]`
where, matching PyTorch `nn.ConvTranspose1d` exactly:

## How It Works

PyTorch parity: the weight layout is `[C_in, C_out, kernelSize]` (the
transposed-convolution convention — input channels first, opposite of the
forward `Conv1DLayer`'s `[C_out, C_in, K]`), and the
`T_out` formula above is bit-identical to `nn.ConvTranspose1d`.

Implemented by delegating to `Engine.ConvTranspose2D` with the time axis
expanded to a degenerate 2D layout — input `[B, C, T]` is reshaped to
`[B, C, 1, T]`, kernel shape is `[C_in, C_out, 1, kernelSize]`,
stride is `(1, stride)`, padding `(0, padding)`, output padding
`(0, outputPadding)`. This reuses the engine's transposed-conv kernel
(including the fused GPU path) and keeps the tape autodiff backward identical
to `DeconvolutionalLayer` — no hand-written backward needed.
We exceed the stock PyTorch op by routing through the engine's fused
conv-transpose + bias (+ activation) kernel when available.

Used by `LayerHelper.CreateDefaultHiFiGANLayers`: each upsample stage is a
`ConvTranspose1d(ch, ch/2, kernel=2*rate, stride=rate, padding=rate/2)`
matching the official `jik876/hifi-gan` generator
(`upsample_rates=[8,8,2,2]`, `upsample_kernel_sizes=[16,16,4,4]`).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Conv1DTransposeLayer(Int32,Int32,Int32,Int32,Nullable<Int32>,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Eager-init constructor — pre-allocates kernel/bias at construction when the input channel count is known up-front (the HiFi-GAN generator stack has fixed per-stage channel counts), so `ParameterCount` and `GetParameters` agree before the fi… |
| `Conv1DTransposeLayer(Int32,Int32,Int32,Nullable<Int32>,Int32,Int32,IActivationFunction<>,IInitializationStrategy<>)` | Lazy-input-channel constructor (mirrors PyTorch's lazy conv semantics). |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Live parameter count: `(C_in·C_out·K) + C_out` once input channels are resolved; before that, falls back to a 1-input-channel estimate so a freshly-constructed model still reports a non-zero `ParameterCount`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeOutputLength(Int32)` | PyTorch `nn.ConvTranspose1d` output-length formula. |
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Serialization metadata — the transposed-conv hyper-parameters aren't recoverable from input/output shapes, so they round-trip here for `CreateLayerFromType` to rebuild an identically-shaped layer on Clone/Deserialize. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

