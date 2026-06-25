---
title: "IOnlineLearningModel<T>"
description: "Defines the interface for online (incremental) learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the interface for online (incremental) learning models.

## For Beginners

Online learning is like learning continuously from experience:

Traditional (Batch) Learning:

- Collect ALL the data first
- Train the model once on everything
- If new data arrives, retrain from scratch

Online (Incremental) Learning:

- Start with minimal or no data
- Learn from each new example as it arrives
- Continuously adapt to new patterns

Why use online learning?

- Streaming data: Data arrives continuously (e.g., stock prices, web clicks)
- Large datasets: Too big to fit in memory all at once
- Changing patterns: Data distribution shifts over time (concept drift)
- Real-time adaptation: Need to respond quickly to new information

Common applications:

- Spam filtering (adapt to new spam patterns)
- Recommendation systems (adapt to user preferences)
- Fraud detection (adapt to new fraud patterns)
- Stock trading (adapt to market conditions)

References:

- Bottou, L. (2010). "Large-Scale Machine Learning with Stochastic Gradient Descent"
- Domingos & Hulten (2000). "Mining High-Speed Data Streams"

## How It Works

Online learning models can update their parameters incrementally as new data arrives,
without needing to retrain from scratch on all data. This is essential for streaming
data and large-scale machine learning.

## Methods

| Method | Summary |
|:-----|:--------|
| `GetLearningRate` | Gets the current learning rate. |
| `GetSampleCount` | Gets the number of samples the model has seen. |
| `PartialFit(Matrix<>,Vector<>)` | Updates the model with a mini-batch of training examples. |
| `PartialFit(Vector<>,)` | Updates the model with a single training example. |
| `PredictSingle(Vector<>)` | Predicts the target value for a single sample. |
| `Reset` | Resets the model to its initial state. |

