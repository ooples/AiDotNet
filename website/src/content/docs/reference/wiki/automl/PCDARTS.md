---
title: "PCDARTS<T>"
description: "Partial Channel Connections for Memory-Efficient Differentiable Architecture Search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AutoML.NAS`

Partial Channel Connections for Memory-Efficient Differentiable Architecture Search.
PC-DARTS reduces memory consumption by sampling only a subset of channels during the search,
making it more scalable to larger search spaces and datasets.

Reference: "PC-DARTS: Partial Channel Connections for Memory-Efficient Architecture Search" (ICLR 2020)

## For Beginners

PC-DARTS makes architecture search more memory-efficient
by only using a subset of channels during the search phase. Regular DARTS uses all
channels which requires huge GPU memory. PC-DARTS samples partial channels, like
tasting a few spoonfuls from a pot instead of drinking the whole thing to judge
the flavor. This enables searching on larger datasets and bigger architectures.

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyEdgeNormalization(Matrix<>)` | Applies edge normalization to prevent operation collapse |
| `DeriveArchitecture` | Derives the discrete architecture |
| `GetArchitectureGradients` | Gets architecture gradients |
| `GetArchitectureParameters` | Gets architecture parameters for optimization |
| `GetChannelSamplingRatio` | Gets the channel sampling ratio |
| `GetMemorySavingsRatio` | Gets the memory savings ratio compared to standard DARTS |
| `SampleChannels(Int32)` | Samples a subset of channels for partial channel connections |

