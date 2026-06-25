---
title: "ActiveLearnerConfig<T>"
description: "Comprehensive configuration for active learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.ActiveLearning.Config`

Comprehensive configuration for active learning.
All properties are nullable - null values use industry-standard defaults.

## For Beginners

Active learning is a machine learning paradigm where the algorithm
actively selects which data points should be labeled by an oracle (human expert). This is
particularly useful when labeling data is expensive or time-consuming.

## How It Works

**How Active Learning Works:**

**Key Concepts:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchBaldCandidates` | Number of candidates to consider for BatchBALD selection. |
| `BatchBaldGreedy` | Whether to use greedy approximation for BatchBALD. |
| `ColdStart` | Strategy for selecting initial labeled samples. |
| `CommitteeSize` | Number of models in the committee for QBC. |
| `CoresetDistance` | Distance metric for coreset selection. |
| `CoresetGreedy` | Whether to use greedy k-center algorithm. |
| `DisagreementMeasure` | How to measure disagreement in the committee. |
| `DiversityClustering` | Clustering method for diversity-based sampling. |
| `DiversityWeight` | Weight for diversity in hybrid uncertainty-diversity strategies. |
| `EnableAutoStop` | Whether to enable automatic stopping before budget exhaustion. |
| `EpochsPerIteration` | Number of training epochs per active learning iteration. |
| `EvaluatePerIteration` | Whether to evaluate on a held-out test set after each iteration. |
| `ExpectedErrorReduction` | Use expected error reduction for query selection. |
| `GradientMethod` | Gradient approximation method for expected model change. |
| `HandleLabelNoise` | Whether to handle label noise in the oracle's responses. |
| `InitialPoolSize` | Number of initially labeled samples to start with. |
| `LearningRate` | Learning rate for model training. |
| `MaxBudget` | Maximum total samples that can be labeled (labeling budget). |
| `McDropoutRate` | Dropout rate for Monte Carlo Dropout. |
| `McDropoutSamples` | Number of Monte Carlo Dropout forward passes for BALD. |
| `MinAccuracyGain` | Minimum accuracy gain required to continue. |
| `QueryBatchSize` | Number of samples to query in each active learning iteration. |
| `QueryStrategy` | Primary query strategy to use for sample selection. |
| `Seed` | Random seed for reproducibility. |
| `StoppingCriterion` | Type of stopping criterion to use. |
| `StoppingPatience` | Patience for stopping criteria (number of iterations without improvement). |
| `StratifiedInitial` | Whether to use stratified sampling for initial selection. |
| `TestSetFraction` | Fraction of data to use as held-out test set. |
| `TrainingBatchSize` | Batch size for training. |
| `UncertaintyMeasure` | Uncertainty measure for uncertainty-based sampling. |
| `WarmStart` | Enable warm starting between iterations (keep model weights). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetEffectiveInitialPoolSize` | Gets the effective initial pool size with default fallback. |
| `GetEffectiveLearningRate` | Gets the effective learning rate with default fallback. |
| `GetEffectiveMaxBudget` | Gets the effective max budget with default fallback. |
| `GetEffectiveQueryBatchSize` | Gets the effective query batch size with default fallback. |
| `GetEffectiveQueryStrategy` | Gets the effective query strategy with default fallback. |

