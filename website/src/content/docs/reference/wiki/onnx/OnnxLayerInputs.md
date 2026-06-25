---
title: "OnnxLayerInputs"
description: "Named tensors flowing INTO a layer's ONNX node(s)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx`

Named tensors flowing INTO a layer's ONNX node(s). The names refer to tensors that
have already been added to the `OnnxGraphBuilder` by an upstream layer
(or the model input). A layer converter consumes these names as the `inputs` field
of the ONNX nodes it emits.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OnnxLayerInputs(IReadOnlyList<String>)` | Named tensors flowing INTO a layer's ONNX node(s). |
| `OnnxLayerInputs(String)` | Convenience constructor for the common single-input case. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HasMultiple` | True if this layer has more than one input tensor (e.g., a concat or residual add). |
| `Primary` | The primary input tensor name (the first entry). |

