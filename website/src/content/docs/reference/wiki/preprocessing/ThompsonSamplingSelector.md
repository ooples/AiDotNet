---
title: "ThompsonSamplingSelector<T>"
description: "Thompson Sampling based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bandit`

Thompson Sampling based Feature Selection.

## For Beginners

Thompson Sampling treats each feature as a "slot machine"
with unknown payout. Instead of tracking a single estimate, we maintain a probability
distribution (Beta distribution) representing our belief about each feature's quality.
We sample from these distributions and pick the feature with the highest sample.
This naturally balances exploration (trying uncertain features) and exploitation
(using features we know are good).

## How It Works

Selects features using Thompson Sampling, a Bayesian approach to the
multi-armed bandit problem that maintains probability distributions over
feature quality.

## Methods

| Method | Summary |
|:-----|:--------|
| `SampleBeta(Double,Double,Random)` | Sample from a Beta(alpha, beta) distribution using the ratio of Gamma samples. |
| `SampleGamma(Double,Random)` | Sample from a Gamma(shape, 1) distribution using Marsaglia-Tsang method. |
| `SampleStandardNormal(Random)` | Sample from standard normal distribution using Box-Muller transform. |

