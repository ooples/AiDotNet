---
title: "HiFiGANResBlockLayer<T>"
description: "HiFi-GAN Multi-Receptive Field (MRF) fusion module (Kong et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

HiFi-GAN Multi-Receptive Field (MRF) fusion module (Kong et al. 2020, §2.2) for
1-D waveform/feature data `[B, C, T]`. After each upsampling stage the
generator runs the input through several residual blocks with different kernel
sizes and dilation patterns IN PARALLEL and returns their averaged sum, so the
network observes patterns over diverse receptive fields simultaneously:

## How It Works

The official `jik876/hifi-gan` v1 config uses
`resblock_kernel_sizes=[3,7,11]` and
`resblock_dilation_sizes=[[1,3,5],[1,3,5],[1,3,5]]` — the defaults here. The
parallel-branch SUM is the defining MRF behaviour; a single sequential dilated-conv
chain is NOT MRF.

Built from inner `Conv1DLayer` instances (two per dilation per
kernel — the leaky-pre-activated dilated conv and the dilation-1 projection), so
the tape handles backward and the fused conv kernels are reused. "Same" padding
keeps T constant across the block (required for the per-branch residual adds and
the cross-branch sum). Reconstructable from `(channels, kernelSizes, dilations)`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HiFiGANResBlockLayer(Int32,Int32[],Int32[])` | Constructs a HiFi-GAN MRF block over the given kernel sizes / dilations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` | Serialization metadata — the block is fully reconstructable from these. |
| `GetParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |
| `UpdateParameters()` |  |

