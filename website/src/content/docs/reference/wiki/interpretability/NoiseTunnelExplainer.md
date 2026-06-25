---
title: "NoiseTunnelExplainer<T, TExplanation>"
description: "Noise Tunnel wrapper that smooths attributions by averaging over noisy inputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

Noise Tunnel wrapper that smooths attributions by averaging over noisy inputs.

## For Beginners

Gradient-based attribution methods can produce noisy results
because gradients are sensitive to small input perturbations. NoiseTunnel addresses
this by averaging attributions computed on multiple noisy versions of the input.

**How it works:**

1. Take the original input
2. Create multiple copies with small random noise added
3. Compute attributions for each noisy copy
4. Average (or aggregate) the attributions

**Aggregation methods:**

- SmoothGrad: Simple average of attributions (most common)
- SmoothGrad-Squared: Average of squared attributions (emphasizes important features)
- VarGrad: Variance of attributions (shows where gradients vary most)

**Benefits:**

- Reduces noise in attribution maps
- Makes explanations more visually interpretable
- More stable under small input changes

**Parameters:**

- NumSamples: More samples = smoother results (default: 5-10)
- StdDev: Standard deviation of noise (typical: 0.1-0.3 of input range)

**Reference:**
Smilkov et al., "SmoothGrad: removing noise by adding noise" (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoiseTunnelExplainer(ILocalExplainer<,>,Func<,Vector<>>,Func<Vector<>,,>,NoiseTunnelType,Int32,Double,Nullable<Int32>)` | Initializes a NoiseTunnel wrapper around a base explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MethodName` | Gets the name of this method. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddGaussianNoise(Vector<>)` | Adds Gaussian noise to an input vector. |
| `AggregateAttributions(List<Vector<>>)` | Aggregates multiple attribution vectors based on the tunnel type. |
| `ComputeMean(List<Vector<>>)` | Computes element-wise mean of attribution vectors. |
| `ComputeSquaredMean(List<Vector<>>)` | Computes element-wise mean of squared attributions. |
| `ComputeStandardDeviation(List<Vector<>>)` | Computes element-wise standard deviation of attributions. |
| `Explain(Vector<>)` | Explains an input with noise-smoothed attributions. |
| `ExplainBatch(Matrix<>)` | Explains multiple inputs with noise-smoothed attributions. |
| `GenerateGaussian` | Generates a sample from standard normal distribution. |

