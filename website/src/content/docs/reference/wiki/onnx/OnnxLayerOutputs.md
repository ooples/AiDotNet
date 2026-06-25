---
title: "OnnxLayerOutputs"
description: "Named tensors flowing OUT of a layer's ONNX node(s)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx`

Named tensors flowing OUT of a layer's ONNX node(s). The names are the tensors a
layer's ONNX nodes write to; downstream layers will reference them as inputs.
The graph builder uses these names to wire the next layer's `OnnxLayerInputs`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnnxLayerOutputs(IReadOnlyList<String>)` | Named tensors flowing OUT of a layer's ONNX node(s). |
| `OnnxLayerOutputs(String)` | Convenience constructor for the common single-output case. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasMultiple` | True if this layer produces more than one output tensor (e.g., a split or a BN training-mode emit). |
| `Primary` | The primary output tensor name (the first entry). |

