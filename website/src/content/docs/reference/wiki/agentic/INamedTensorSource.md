---
title: "INamedTensorSource"
description: "A source of named weight tensors readable as `Double` arrays — the common surface over the safetensors and GGUF readers that `WeightImporter` imports from."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models.Local`

A source of named weight tensors readable as `Double` arrays — the common surface over the
safetensors and GGUF readers that `WeightImporter` imports from.

## For Beginners

Both file formats (safetensors, GGUF) can list their weight arrays by name and
hand them back as numbers. This interface is that shared ability, so the importer doesn't care which format
the weights came from.

## Properties

| Property | Summary |
|:-----|:--------|
| `TensorNames` | Gets the names of the tensors available in this source. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ReadAsDouble(String)` | Reads a tensor's values as `Double`. |

