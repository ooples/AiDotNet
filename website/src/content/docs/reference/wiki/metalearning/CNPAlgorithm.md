---
title: "CNPAlgorithm<T, TInput, TOutput>"
description: "Implementation of Conditional Neural Process (CNP) (Garnelo et al., ICML 2018)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Conditional Neural Process (CNP) (Garnelo et al., ICML 2018).

## How It Works

CNP encodes each context pair independently, aggregates via mean pooling, and decodes
to predict target values. It provides amortized inference without gradient-based adaptation.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

