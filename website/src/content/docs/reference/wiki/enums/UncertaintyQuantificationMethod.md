---
title: "UncertaintyQuantificationMethod"
description: "Defines the supported uncertainty quantification strategies for inference."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the supported uncertainty quantification strategies for inference.

## For Beginners

Some models can also tell you how confident they are.
This enum lets you choose the strategy used to estimate that confidence.

## How It Works

These options control how the system estimates predictive uncertainty during inference.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | Automatically selects a suitable method when possible, otherwise falls back to deterministic predictions. |
| `BayesianNeuralNetwork` | Uses Bayesian neural network sampling (e.g., Bayes-by-Backprop style layers) to estimate uncertainty. |
| `ConformalPrediction` | Uses conformal prediction to produce statistically valid intervals (regression) or prediction sets (classification). |
| `DeepEnsemble` | Uses a deep ensemble (multiple independently trained models) to estimate uncertainty. |
| `LaplaceApproximation` | Uses a Laplace approximation (typically diagonal) over model parameters to sample predictions. |
| `MonteCarloDropout` | Uses Monte Carlo Dropout by enabling dropout at inference and sampling multiple forward passes. |
| `Swag` | Uses SWAG (Stochastic Weight Averaging-Gaussian) to sample model parameters and estimate uncertainty. |

