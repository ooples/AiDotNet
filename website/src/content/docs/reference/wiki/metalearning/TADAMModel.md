---
title: "TADAMModel<T, TInput, TOutput>"
description: "TADAM model for few-shot classification with task conditioning and metric scaling."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Models`

TADAM model for few-shot classification with task conditioning and metric scaling.

## For Beginners

After TADAM sees the support examples and computes
task-conditioned prototypes, this model stores those prototypes along with the
learned metric scaling parameters. It can then classify new examples by measuring
scaled distances to these prototypes.

## How It Works

This model stores the adapted state of TADAM after computing task-conditioned prototypes.
It uses learned metric scaling and temperature to classify new query examples.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TADAMModel(IFullModel<,,>,Dictionary<Int32,Tensor<>>,Vector<>,,TADAMOptions<,,>)` | Initializes a new instance of the TADAMModel. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Metadata` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplySoftmax(Vector<>)` | Applies softmax to convert logits to probabilities. |
| `ComputeLogits(Vector<>)` | Converts distances to logits using temperature scaling. |
| `ComputeScaledDistance(Vector<>,Tensor<>)` | Computes the scaled Euclidean distance between query and prototype. |
| `ComputeScaledDistances(Vector<>)` | Computes scaled distances from the query to each class prototype. |
| `ConvertToOutput(Vector<>)` | Converts probability vector to the expected output type. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `Predict()` |  |
| `Train(,)` |  |
| `UpdateParameters(Vector<>)` |  |

