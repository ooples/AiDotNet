---
title: "WaveNetResidualBlockLayer<T>"
description: "A single WaveNet / Parallel WaveGAN residual block (van den Oord et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

A single WaveNet / Parallel WaveGAN residual block (van den Oord et al. 2016;
Yamamoto et al. 2020) for 1-D waveform/feature data `[B, C, T]`.
Implements the paper's gated-activation residual unit:

## How It Works

The dual filter/gate dilated convolutions and the `tanh·sigmoid` product are
the defining WaveNet gated activation — a plain dilated-conv stack (no gating, no
residual) is NOT WaveNet. The residual connection carries the signal forward
through the deep dilation stack exactly as in the paper's residual path.

Built from three inner `Conv1DLayer` instances (filter, gate, 1×1
projection), so the gradient tape and the fused conv kernels are reused — no
hand-written backward. Channel width is constant across the block (the residual
add requires `C_out == C_in`); the block is reconstructable purely from
`(channels, kernelSize, dilation)` for Clone/Deserialize.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `WaveNetResidualBlockLayer(Int32,Int32,Int32)` | Constructs a WaveNet gated-residual block of constant channel width. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Serialization metadata — the block is fully reconstructable from these. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |
| `UpdateParameters()` |  |

