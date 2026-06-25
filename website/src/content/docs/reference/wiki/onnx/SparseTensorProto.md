---
title: "SparseTensorProto"
description: "A serialized sparse-tensor value"
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

A serialized sparse-tensor value

## Properties

| Property | Summary |
|:-----|:--------|
| `Dims` | The shape of the underlying dense-tensor: [dim_1, dim_2, ... |
| `Indices` | The indices of the non-default values, which may be stored in one of two formats. |
| `Values` | The sequence of non-default values are encoded as a tensor of shape [NNZ]. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DimsFieldNumber` | Field number for the "dims" field. |
| `IndicesFieldNumber` | Field number for the "indices" field. |
| `ValuesFieldNumber` | Field number for the "values" field. |

