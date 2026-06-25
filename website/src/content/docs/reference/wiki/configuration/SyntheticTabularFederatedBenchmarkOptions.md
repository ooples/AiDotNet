---
title: "SyntheticTabularFederatedBenchmarkOptions"
description: "Configuration options for the synthetic federated tabular benchmark suite."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for the synthetic federated tabular benchmark suite.

## For Beginners

Synthetic means AiDotNet generates the data automatically instead of reading
files from disk. This is useful for CI and quick sanity checks.

## How It Works

This suite generates a deterministic synthetic dataset with non-IID client distributions, allowing
benchmark runs without external dataset dependencies.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClassCount` | Gets or sets the number of classes for classification tasks (null uses defaults). |
| `ClientCount` | Gets or sets the number of federated clients to simulate (null uses defaults). |
| `DirichletAlpha` | Gets or sets the Dirichlet concentration parameter controlling label skew across clients. |
| `FeatureCount` | Gets or sets the number of input features per sample (null uses defaults). |
| `NoiseStdDev` | Gets or sets the noise standard deviation applied to generated targets/scores. |
| `TaskType` | Gets or sets the synthetic task type. |
| `TestSamplesPerClient` | Gets or sets the number of test samples per client (null uses defaults). |
| `TrainSamplesPerClient` | Gets or sets the number of training samples per client (null uses defaults). |

