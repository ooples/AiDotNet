---
title: "DynamicShapeInfo"
description: "Describes which dimensions of a model's input/output shapes are dynamic (variable at runtime)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Describes which dimensions of a model's input/output shapes are dynamic (variable at runtime).

## How It Works

**For Beginners:** Most machine learning models have fixed input sizes (e.g., always 784 features).
But some models can handle variable-sized inputs. For example, a model might accept any batch size,
or a sequence model might handle different sequence lengths. This class describes which dimensions
can vary and what their valid ranges are.

This follows the ONNX convention where -1 in a shape dimension means "variable at runtime".
For example, an input shape of [-1, 784] means "any number of samples, each with 784 features".

Industry standards:

- ONNX: Uses -1 for dynamic dimensions in shape arrays
- TensorFlow Serving: Uses None/null for dynamic dimensions
- TorchServe: Uses -1 for dynamic dimensions

## Properties

| Property | Summary |
|:-----|:--------|
| `DynamicInputDimensions` | Gets the indices of input dimensions that are variable at runtime. |
| `DynamicOutputDimensions` | Gets the indices of output dimensions that are variable at runtime. |
| `HasDynamicBatch` | Gets whether the batch dimension (index 0) is dynamic. |
| `HasDynamicInput` | Gets whether any input dimensions are dynamic. |
| `HasDynamicOutput` | Gets whether any output dimensions are dynamic. |
| `MaxInputDimensions` | Gets the maximum allowed values for dynamic input dimensions. |
| `MinInputDimensions` | Gets the minimum allowed values for dynamic input dimensions. |

## Methods

| Method | Summary |
|:-----|:--------|
| `IsValidShape(Int32[],Int32[],Int32[],Int32[],Int32[])` | Validates a concrete shape against a template shape, respecting dynamic dimensions. |

## Fields

| Field | Summary |
|:-----|:--------|
| `None` | A shared instance representing no dynamic dimensions (all dimensions are fixed). |

