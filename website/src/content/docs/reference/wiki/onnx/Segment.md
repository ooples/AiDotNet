---
title: "Segment"
description: "For very large tensors, we may want to store them in chunks, in which case the following fields will specify the segment that is stored in the current TensorProto."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

For very large tensors, we may want to store them in chunks, in which
case the following fields will specify the segment that is stored in
the current TensorProto.

## Fields

| Field | Summary |
|:-----|:--------|
| `BeginFieldNumber` | Field number for the "begin" field. |
| `EndFieldNumber` | Field number for the "end" field. |

