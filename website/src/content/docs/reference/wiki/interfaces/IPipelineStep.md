---
title: "IPipelineStep<T, TInput, TOutput>"
description: "Represents a step in a data processing pipeline"
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a step in a data processing pipeline

## For Beginners

A pipeline step is a modular component that processes data in stages.
Each step can fit (learn from data), transform (process data), or both. This pattern allows you to
chain multiple processing steps together to create complex data processing workflows.

## How It Works

The generic parameters allow this interface to work with different types of data while maintaining
type safety. T is typically a numeric type (like double or float) used for calculations, while TInput
and TOutput define what types of data the step accepts and produces.

## Methods

| Method | Summary |
|:-----|:--------|
| `FitAsync(,)` | Fits/trains this pipeline step on the provided data |
| `FitTransformAsync(,)` | Fits and transforms in a single operation (convenience method) |
| `GetMetadata` | Gets metadata about this pipeline step |
| `GetParameters` | Gets the parameters of this pipeline step |
| `SetParameters(Dictionary<String,Object>)` | Sets the parameters of this pipeline step |
| `TransformAsync()` | Transforms the input data using the fitted model |
| `ValidateInput()` | Validates that this step can process the given input |

