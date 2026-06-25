---
title: "PipelineStep<T, TInput>"
description: "Represents a named step in a preprocessing pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing`

Represents a named step in a preprocessing pipeline.

## How It Works

This class replaces ValueTuples for pipeline steps to ensure proper JSON serialization.
ValueTuples do not serialize reliably with Newtonsoft.Json, so this class provides
explicit property names and serialization attributes.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PipelineStep` | Creates a new pipeline step. |
| `PipelineStep(String,IDataTransformer<,,>)` | Creates a new pipeline step with the specified name and transformer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets or sets the name of this pipeline step. |
| `Transformer` | Gets or sets the transformer for this pipeline step. |

