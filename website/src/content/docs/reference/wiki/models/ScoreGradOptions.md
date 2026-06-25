---
title: "ScoreGradOptions<T>"
description: "Configuration options for ScoreGrad (Score-based Gradient Models for Time Series)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for ScoreGrad (Score-based Gradient Models for Time Series).

## For Beginners

ScoreGrad uses a different approach to diffusion models
called "score matching":

**The Key Insight:**
Instead of learning to predict noise directly, ScoreGrad learns the "score" -
the direction that points toward higher probability regions. Following the score
gradient leads the model from noise toward realistic time series.

**What is the Score Function?**
The score is the gradient of the log probability: ∇_x log p(x).

- It points toward regions of high probability
- Following it uphill finds the most likely data
- Denoising Score Matching (DSM) provides a way to learn it

**How ScoreGrad Works:**

1. **Score Network:** Train a network to predict ∇_x log p(x|σ) for various noise levels
2. **Noise Conditioning:** Condition on noise level σ for multi-scale scores
3. **Langevin Dynamics:** Use stochastic gradient ascent to sample from the distribution
4. **Annealed Sampling:** Start with high noise, gradually reduce for refinement

**ScoreGrad Architecture:**

- Score Network: Predicts gradient direction at each noise level
- Noise Embedding: Encodes current noise level for conditioning
- Time Embedding: Encodes temporal position information
- Skip Connections: Preserves input details during score computation

**Key Benefits:**

- Principled probabilistic foundation (score matching)
- Flexible noise schedules
- Can use Langevin dynamics for sampling
- Works well for time series with complex dynamics

## How It Works

ScoreGrad is a score-based generative model for time series that learns the gradient
of the log probability density (score function) for denoising and generation.

**Reference:** Yan et al., "ScoreGrad: Multivariate Probabilistic Time Series Forecasting with Continuous Energy-based Generative Models", 2021.
https://arxiv.org/abs/2106.10121

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ScoreGradOptions` | Initializes a new instance of the `ScoreGradOptions` class with default values. |
| `ScoreGradOptions(ScoreGradOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AnnealingPower` | Gets or sets the annealing power for noise schedule. |
| `DropoutRate` | Gets or sets the dropout rate. |
| `ForecastHorizon` | Gets or sets the forecast horizon. |
| `HiddenDimension` | Gets or sets the hidden dimension of the score network. |
| `NumFeatures` | Gets or sets the number of features. |
| `NumLangevinSteps` | Gets or sets the number of Langevin dynamics steps per noise level. |
| `NumLayers` | Gets or sets the number of layers in the score network. |
| `NumNoiseScales` | Gets or sets the number of noise scales (sigma levels). |
| `NumSamples` | Gets or sets the number of samples for uncertainty estimation. |
| `SequenceLength` | Gets or sets the sequence length (context length). |
| `SigmaMax` | Gets or sets the maximum noise level (sigma). |
| `SigmaMin` | Gets or sets the minimum noise level (sigma). |
| `StepSize` | Gets or sets the step size (epsilon) for Langevin dynamics. |
| `UseAnnealing` | Gets or sets whether to use annealed Langevin sampling. |

