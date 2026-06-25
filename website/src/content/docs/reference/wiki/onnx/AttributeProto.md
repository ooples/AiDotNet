---
title: "AttributeProto"
description: "Attributes  A named attribute containing either singular float, integer, string, graph, and tensor values, or repeated float, integer, string, graph, and tensor values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Attributes

A named attribute containing either singular float, integer, string, graph,
and tensor values, or repeated float, integer, string, graph, and tensor values.
An AttributeProto MUST contain the name field, and *only one* of the
following content fields, effectively enforcing a C/C++ union equivalent.

## Properties

| Property | Summary |
|:-----|:--------|
| `DocString` | A human-readable documentation for this attribute. |
| `F` | Exactly ONE of the following fields must be present for this version of the IR |
| `Floats` | list of floats |
| `G` | graph |
| `Graphs` | list of graph |
| `I` | int |
| `Ints` | list of ints |
| `Name` | The name field MUST be present for this version of the IR. |
| `RefAttrName` | if ref_attr_name is not empty, ref_attr_name is the attribute name in parent function. |
| `S` | UTF-8 string |
| `SparseTensor` | sparse tensor value |
| `SparseTensors` | list of sparse tensors |
| `Strings` | list of UTF-8 strings |
| `T` | tensor value |
| `Tensors` | list of tensors |
| `Tp` | Do not use field below, it's deprecated. |
| `Type` | The type field MUST be present for this version of the IR. |
| `TypeProtos` | list of type protos |

## Fields

| Field | Summary |
|:-----|:--------|
| `DocStringFieldNumber` | Field number for the "doc_string" field. |
| `FFieldNumber` | Field number for the "f" field. |
| `FloatsFieldNumber` | Field number for the "floats" field. |
| `GFieldNumber` | Field number for the "g" field. |
| `GraphsFieldNumber` | Field number for the "graphs" field. |
| `IFieldNumber` | Field number for the "i" field. |
| `IntsFieldNumber` | Field number for the "ints" field. |
| `NameFieldNumber` | Field number for the "name" field. |
| `RefAttrNameFieldNumber` | Field number for the "ref_attr_name" field. |
| `SFieldNumber` | Field number for the "s" field. |
| `SparseTensorFieldNumber` | Field number for the "sparse_tensor" field. |
| `SparseTensorsFieldNumber` | Field number for the "sparse_tensors" field. |
| `StringsFieldNumber` | Field number for the "strings" field. |
| `TFieldNumber` | Field number for the "t" field. |
| `TensorsFieldNumber` | Field number for the "tensors" field. |
| `TpFieldNumber` | Field number for the "tp" field. |
| `TypeFieldNumber` | Field number for the "type" field. |
| `TypeProtosFieldNumber` | Field number for the "type_protos" field. |

