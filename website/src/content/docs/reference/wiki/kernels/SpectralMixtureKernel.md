---
title: "SpectralMixtureKernel<T>"
description: "Implements the Spectral Mixture (SM) kernel for discovering and exploiting patterns in data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Spectral Mixture (SM) kernel for discovering and exploiting patterns in data.

## For Beginners

The Spectral Mixture kernel is a highly flexible kernel that can learn
complex patterns from data, including multiple periodicities at different scales.

Key idea: Any stationary kernel can be expressed as a mixture of cosine functions
(this is Bochner's theorem). The SM kernel makes this explicit by parameterizing
a mixture of Gaussians in the frequency domain.

In mathematical terms:
k(τ) = Σᵢ wᵢ × exp(-2π²τ²σᵢ²) × cos(2πτμᵢ)

Where:

- τ = |x - x'| is the distance between points
- wᵢ is the weight (importance) of component i
- μᵢ is the frequency (how fast the pattern repeats)
- σᵢ is the bandwidth (how wide the peak is in frequency domain)

## How It Works

Why use Spectral Mixture?

1. **Pattern Discovery**: Automatically finds periodicities in data
2. **Multiple Scales**: Can capture patterns at different frequencies
3. **Interpretability**: Components correspond to different pattern types
4. **Flexibility**: Can approximate any stationary kernel

Examples:

- Stock prices: Daily, weekly, monthly, yearly patterns
- Weather data: Daily and seasonal cycles
- Audio signals: Multiple frequency components

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralMixtureKernel(Double[],Double[],Double[])` | Initializes a new Spectral Mixture kernel with specified parameters. |
| `SpectralMixtureKernel(Int32,Double)` | Initializes a new Spectral Mixture kernel with default initialization. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumComponents` | Gets the number of mixture components. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Spectral Mixture kernel value between two vectors. |
| `EstimateFrequencies(Double[],Int32)` | Estimates initial frequencies from data using spectral analysis. |
| `GetBandwidths` | Gets a copy of the component bandwidths. |
| `GetFrequencies` | Gets a copy of the component frequencies. |
| `GetWeights` | Gets a copy of the component weights. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bandwidths` | The bandwidths (length scales) for each mixture component. |
| `_frequencies` | The frequencies (means) for each mixture component. |
| `_numComponents` | The number of mixture components. |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_weights` | The weights for each mixture component. |

