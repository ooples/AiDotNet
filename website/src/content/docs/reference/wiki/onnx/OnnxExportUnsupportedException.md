---
title: "OnnxExportUnsupportedException"
description: "Thrown when a model contains a layer or component that does not yet have an ONNX export converter."
section: "API Reference"
---

`Exceptions` · `AiDotNet.Onnx`

Thrown when a model contains a layer or component that does not yet have an ONNX
export converter. The exception message names the unsupported component so callers
can point users at the right follow-up (open an issue, request the converter, or
switch to a supported layer type).

See `OnnxSupportMatrix` (forthcoming doc) for the
canonical list of supported layer types.

## Properties

| Property | Summary |
|:-----|:--------|
| `ComponentTypeName` | The type name of the unsupported component (e.g., "ConvolutionalLayer"). |

