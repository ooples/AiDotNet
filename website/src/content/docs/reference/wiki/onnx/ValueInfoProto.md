---
title: "ValueInfoProto"
description: "Defines information on value, including the name, the type, and the shape of the value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Defines information on value, including the name, the type, and
the shape of the value.

## Properties

| Property | Summary |
|:-----|:--------|
| `DocString` | A human-readable documentation for this value. |
| `MetadataProps` | Named metadata values; keys should be distinct. |
| `Name` | This field MUST be present in this version of the IR. |
| `Type` | This field MUST be present in this version of the IR for inputs and outputs of the top-level graph. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DocStringFieldNumber` | Field number for the "doc_string" field. |
| `MetadataPropsFieldNumber` | Field number for the "metadata_props" field. |
| `NameFieldNumber` | Field number for the "name" field. |
| `TypeFieldNumber` | Field number for the "type" field. |

