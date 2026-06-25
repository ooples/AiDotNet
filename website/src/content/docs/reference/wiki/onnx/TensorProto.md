---
title: "TensorProto"
description: "Tensors  A serialized tensor value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Tensors

A serialized tensor value.

## Properties

| Property | Summary |
|:-----|:--------|
| `DataLocation` | If value not set, data is stored in raw_data (if set) otherwise in type-specified field. |
| `DataType` | The data type of the tensor. |
| `Dims` | The shape of the tensor. |
| `DocString` | A human-readable documentation for this tensor. |
| `DoubleData` | For double Complex128 tensors are encoded as a single array of doubles, with the real components appearing in odd numbered positions, and the corresponding imaginary component appearing in the subsequent even numbered position. |
| `ExternalData` | Data can be stored inside the protobuf file using type-specific fields or raw_data. |
| `FloatData` | For float and complex64 values Complex64 tensors are encoded as a single array of floats, with the real components appearing in odd numbered positions, and the corresponding imaginary component appearing in the subsequent even numbered posi… |
| `Int32Data` | For int32, uint8, int8, uint16, int16, uint4, int4, bool, float8 and float16 values float16 and float8 values must be bit-wise converted to an uint16_t prior to writing to the buffer. |
| `Int64Data` | For int64. |
| `MetadataProps` | Named metadata values; keys should be distinct. |
| `Name` | Optionally, a name for the tensor. |
| `RawData` | Serializations can either use one of the fields above, or use this raw bytes field. |
| `StringData` | For strings. |
| `Uint64Data` | For uint64 and uint32 values When this field is present, the data_type field MUST be UINT32 or UINT64 |

## Fields

| Field | Summary |
|:-----|:--------|
| `DataLocationFieldNumber` | Field number for the "data_location" field. |
| `DataTypeFieldNumber` | Field number for the "data_type" field. |
| `DimsFieldNumber` | Field number for the "dims" field. |
| `DocStringFieldNumber` | Field number for the "doc_string" field. |
| `DoubleDataFieldNumber` | Field number for the "double_data" field. |
| `ExternalDataFieldNumber` | Field number for the "external_data" field. |
| `FloatDataFieldNumber` | Field number for the "float_data" field. |
| `Int32DataFieldNumber` | Field number for the "int32_data" field. |
| `Int64DataFieldNumber` | Field number for the "int64_data" field. |
| `MetadataPropsFieldNumber` | Field number for the "metadata_props" field. |
| `NameFieldNumber` | Field number for the "name" field. |
| `RawDataFieldNumber` | Field number for the "raw_data" field. |
| `SegmentFieldNumber` | Field number for the "segment" field. |
| `StringDataFieldNumber` | Field number for the "string_data" field. |
| `Uint64DataFieldNumber` | Field number for the "uint64_data" field. |

