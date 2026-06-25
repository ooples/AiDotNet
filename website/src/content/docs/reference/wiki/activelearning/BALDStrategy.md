---
title: "BALDStrategy<T, TInput, TOutput>"
description: "Bayesian Active Learning by Disagreement (BALD) strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Strategies.Bayesian`

Bayesian Active Learning by Disagreement (BALD) strategy.

## For Beginners

BALD uses Bayesian principles to select samples that would
provide the most information about the model's parameters. It measures the mutual
information between predictions and model parameters.

## How It Works

**How BALD Works:**

**When to Use:**

**Reference:** Gal et al. "Deep Bayesian Active Learning with Image Data" (ICML 2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BALDStrategy` | Initializes a new BALD strategy with default parameters. |
| `BALDStrategy(Int32,Double,ActiveLearnerConfig<>)` | Initializes a new BALD strategy with specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `MonteCarloSamples` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeMutualInformation(IFullModel<,,>,)` |  |
| `ComputePredictiveEntropy(IFullModel<,,>,)` |  |
| `ComputeScores(IFullModel<,,>,IDataset<,,>)` |  |
| `Reset` |  |
| `SelectSamples(IFullModel<,,>,IDataset<,,>,Int32)` |  |
| `UpdateState(Int32[],[])` |  |

