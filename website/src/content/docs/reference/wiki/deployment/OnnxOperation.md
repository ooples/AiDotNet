---
title: "OnnxOperation"
description: "Represents an ONNX operation (node in the computational graph)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Deployment.Export.Onnx`

Represents an ONNX operation (node in the computational graph).

## For Beginners

OnnxOperation provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attributes` | Gets the operation attributes (parameters). |
| `Domain` | Gets or sets the domain (for custom operators). |
| `Inputs` | Gets the list of input names. |
| `Name` | Gets or sets the operation name. |
| `Outputs` | Gets the list of output names. |
| `Type` | Gets or sets the operation type (e.g., "Conv", "Relu", "MatMul"). |

