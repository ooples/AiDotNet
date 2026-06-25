---
title: "ContinualLearningOptions"
description: "Represents configuration options for continual learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Represents configuration options for continual learning.

## For Beginners

Continual learning (also called lifelong learning) allows models
to learn from a sequence of tasks without forgetting what was learned before. This is important
because standard neural networks suffer from "catastrophic forgetting" - when trained on new
data, they tend to forget previously learned patterns.

## How It Works

**Typical Usage:**

**How to Choose a Strategy:**

## Properties

| Property | Summary |
|:-----|:--------|
| `BufferStrategy` | Gets or sets the buffer management strategy for Experience Replay. |
| `Damping` | Gets or sets the damping constant for Synaptic Intelligence. |
| `DropoutRate` | Gets or sets the dropout rate for Monte Carlo Dropout in VCL. |
| `FisherSampleSize` | Gets or sets the number of samples to use for Fisher Information estimation. |
| `Gamma` | Gets or sets the decay factor for Online EWC. |
| `InitialLogVariance` | Gets or sets the initial log-variance for weight distributions in VCL. |
| `Lambda` | Gets or sets the regularization strength (lambda) for weight consolidation strategies. |
| `Margin` | Gets or sets the margin for gradient constraints in GEM. |
| `MaxBufferSize` | Gets or sets the maximum buffer size for Experience Replay. |
| `MemorySize` | Gets or sets the maximum number of samples to store per task in memory-based strategies. |
| `NormalizeImportance` | Gets or sets whether to normalize importance scores. |
| `NumMcSamples` | Gets or sets the number of Monte Carlo samples for VCL. |
| `PruningRatio` | Gets or sets the pruning ratio for PackNet. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `ReplayBatchSize` | Gets or sets the batch size for generating pseudo-examples in Generative Replay. |
| `ReplayRatio` | Gets or sets the ratio of replay samples to new samples in replay-based strategies. |
| `SampleSize` | Gets or sets the batch size to sample from memory for reference gradients in A-GEM. |
| `Strategy` | Gets or sets the continual learning strategy to use. |
| `Temperature` | Gets or sets the temperature for knowledge distillation in LearningWithoutForgetting. |
| `UseLateralConnections` | Gets or sets whether to use lateral connections in Progressive Neural Networks. |

