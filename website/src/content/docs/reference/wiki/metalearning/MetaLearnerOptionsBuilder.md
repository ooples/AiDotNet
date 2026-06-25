---
title: "MetaLearnerOptionsBuilder<T>"
description: "Fluent builder for MetaLearnerOptionsBase."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.MetaLearning`

Fluent builder for MetaLearnerOptionsBase.

## Methods

| Method | Summary |
|:-----|:--------|
| `Build` | Builds the options instance. |
| `WithAdaptationSteps(Int32)` | Sets the number of adaptation steps. |
| `WithCheckpointing(Boolean,Int32)` | Configures checkpointing settings. |
| `WithEvaluation(Int32,Int32)` | Configures evaluation settings. |
| `WithFirstOrder(Boolean)` | Enables first-order approximation (FOMAML). |
| `WithGradientClipping(Nullable<Double>)` | Sets the gradient clipping threshold. |
| `WithInnerLearningRate(Double)` | Sets the inner loop learning rate. |
| `WithMetaBatchSize(Int32)` | Sets the meta-batch size. |
| `WithNumMetaIterations(Int32)` | Sets the number of meta-iterations. |
| `WithOuterLearningRate(Double)` | Sets the outer loop learning rate. |
| `WithRandomSeed(Nullable<Int32>)` | Sets the random seed for reproducibility. |

