---
title: "IPostprocessor<T, TInput, TOutput>"
description: "Defines a postprocessor that transforms model outputs into final results."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines a postprocessor that transforms model outputs into final results.

## For Beginners

A postprocessor is like a translator that:

1. Takes raw model output (like numbers or probabilities)
2. Converts it into something meaningful (like text, labels, or structured data)

Examples:

- Converting softmax outputs to class labels
- Decoding text from token IDs
- Applying Non-Maximum Suppression to bounding boxes
- Cleaning up OCR text output

## How It Works

This is the core interface for all postprocessing operations in AiDotNet.
Unlike preprocessing (which follows sklearn-style Fit/Transform pattern),
postprocessing typically doesn't require fitting - it transforms model outputs
directly into the desired format.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsConfigured` | Gets whether this postprocessor requires configuration before use. |
| `SupportsInverse` | Gets whether this postprocessor supports inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Configure(Dictionary<String,Object>)` | Configures the postprocessor with optional settings. |
| `Inverse()` | Reverses the postprocessing (if supported). |
| `Process()` | Transforms model output into the final result format. |
| `ProcessBatch(IEnumerable<>)` | Transforms a batch of model outputs. |

