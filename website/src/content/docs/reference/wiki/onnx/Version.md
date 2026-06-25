---
title: "Version"
description: "Versioning  ONNX versioning is specified in docs/IR.md and elaborated on in docs/Versioning.md  To be compatible with both proto2 and proto3, we will use a version number that is not defined by the default value but an explicit enum number."
section: "API Reference"
---

`Enums` · `AiDotNet.Onnx.Protobuf`

Versioning

ONNX versioning is specified in docs/IR.md and elaborated on in docs/Versioning.md

To be compatible with both proto2 and proto3, we will use a version number
that is not defined by the default value but an explicit enum number.

## Fields

| Field | Summary |
|:-----|:--------|
| `IrVersion` | IR VERSION 10 published on TBD Added UINT4, INT4. |
| `IrVersion20171010` | The version field is always serialized and we will use it to store the version that the graph is generated from. |
| `IrVersion20171030` | IR_VERSION 2 published on Oct 30, 2017 - Added type discriminator to AttributeProto to support proto3 users |
| `IrVersion2017113` | IR VERSION 3 published on Nov 3, 2017 - For operator versioning: - Added new message OperatorSetIdProto - Added opset_import in ModelProto - For vendor extensions, added domain in NodeProto |
| `IrVersion2019122` | IR VERSION 4 published on Jan 22, 2019 - Relax constraint that initializers should be a subset of graph inputs - Add type BFLOAT16 |
| `IrVersion2019318` | IR VERSION 5 published on March 18, 2019 - Add message TensorAnnotation. |
| `IrVersion2019919` | IR VERSION 6 published on Sep 19, 2019 - Add support for sparse tensor constants stored in model. |
| `IrVersion202058` | IR VERSION 7 published on May 8, 2020 - Add support to allow function body graph to rely on multiple external opreator sets. |
| `IrVersion2021730` | IR VERSION 8 published on July 30, 2021 Introduce TypeProto.SparseTensor Introduce TypeProto.Optional Added a list of FunctionProtos local to the model Deprecated since_version and operator status from FunctionProto |
| `IrVersion202355` | IR VERSION 9 published on May 5, 2023 Added AttributeProto to FunctionProto so that default attribute values can be set. |
| `StartVersion` | proto3 requires the first enum value to be zero. |

