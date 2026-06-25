---
title: "InContextLearning<T>"
description: "Helper class for in-context learning in tabular foundation models like TabPFN."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Helper class for in-context learning in tabular foundation models like TabPFN.

## For Beginners

Think of in-context learning like this:

- Traditional ML: Train a model, then use it to predict
- In-context learning: Give the model examples AND the test data together

The model "learns" from the examples in real-time through attention mechanisms.

## How It Works

In-context learning allows models to learn from examples provided at inference time
without updating model parameters. The model conditions on training examples
to make predictions on new data.

## Properties

| Property | Summary |
|:-----|:--------|
| `HasContext` | Gets whether context has been set. |
| `NumContextSamples` | Gets the number of samples in the current context. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearContext` | Clears the current context. |
| `CreateContextMask(Int32)` | Creates a mask indicating which positions are context vs query. |
| `CreateInContextInput(Tensor<>)` | Creates the combined input for in-context learning by concatenating context samples with query samples. |
| `ExtractQueryOutput(Tensor<>,Int32)` | Extracts query predictions from combined output. |
| `GetContextLabels` | Gets the context labels for use in attention mechanisms. |
| `SetClassificationContext(Tensor<>,Vector<Int32>,Int32,Int32)` | Sets the context for classification tasks with integer labels. |
| `SetContext(Tensor<>,Tensor<>,Int32)` | Sets the context (training) data for in-context learning. |
| `SubsampleContext(Tensor<>,Tensor<>,Int32)` | Subsamples context to fit within maximum size. |

