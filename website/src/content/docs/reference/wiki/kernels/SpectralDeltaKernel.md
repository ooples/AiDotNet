---
title: "SpectralDeltaKernel<T>"
description: "Spectral Delta Kernel representing a single spectral component."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Spectral Delta Kernel representing a single spectral component.

## For Beginners

The Spectral Delta Kernel is a single component of the Spectral Mixture Kernel.
While the SpectralMixtureKernel uses multiple frequency components, this kernel uses just one.

In mathematical terms:
k(τ) = σ² × exp(-2π²τ²σ_f²) × cos(2πμτ)

Where:

- τ = |x - x'| is the distance between points
- σ² is the output variance (weight)
- μ is the frequency (how fast the pattern repeats)
- σ_f is the frequency bandwidth (how "spread" the frequency is)

Think of it as a single "note" in the spectrum:

- μ determines the pitch (frequency)
- σ_f determines how pure the tone is (narrow = pure, wide = noisy)
- σ² determines the volume

This is useful when:

- You expect a single dominant periodicity
- You want a simple periodic pattern with decay
- As a building block for more complex spectral kernels

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SpectralDeltaKernel(Double,Double,Double)` | Initializes a new Spectral Delta Kernel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Bandwidth` | Gets the frequency bandwidth. |
| `Frequency` | Gets the mean frequency. |
| `Period` | Gets the period (1/frequency). |
| `Variance` | Gets the output variance. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the spectral delta kernel value. |
| `FromPeriod(Double,Double,Double)` | Creates a Spectral Delta Kernel from a known period. |
| `GetPowerSpectralDensity(Double)` | Computes the power spectral density at a given frequency. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_bandwidth` | The frequency bandwidth (σ_f). |
| `_frequency` | The mean frequency (μ). |
| `_numOps` | Operations for performing numeric calculations with type T. |
| `_variance` | The output variance (weight). |

