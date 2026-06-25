---
title: "ModelProto"
description: "Models  ModelProto is a top-level file/container format for bundling a ML model and associating its computation graph with metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Models

ModelProto is a top-level file/container format for bundling a ML model and
associating its computation graph with metadata.

The semantics of the model are described by the associated GraphProto's.

## Properties

| Property | Summary |
|:-----|:--------|
| `DocString` | A human-readable documentation for this model. |
| `Domain` | Domain name of the model. |
| `Functions` | A list of function protos local to the model. |
| `Graph` | The parameterized graph that is evaluated to execute the model. |
| `IrVersion` | The version of the IR this model targets. |
| `MetadataProps` | Named metadata values; keys should be distinct. |
| `ModelVersion` | The version of the graph encoded. |
| `OpsetImport` | The OperatorSets this model relies on. |
| `ProducerName` | The name of the framework or tool used to generate this model. |
| `ProducerVersion` | The version of the framework or tool used to generate this model. |
| `TrainingInfo` | Training-specific information. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DocStringFieldNumber` | Field number for the "doc_string" field. |
| `DomainFieldNumber` | Field number for the "domain" field. |
| `FunctionsFieldNumber` | Field number for the "functions" field. |
| `GraphFieldNumber` | Field number for the "graph" field. |
| `IrVersionFieldNumber` | Field number for the "ir_version" field. |
| `MetadataPropsFieldNumber` | Field number for the "metadata_props" field. |
| `ModelVersionFieldNumber` | Field number for the "model_version" field. |
| `OpsetImportFieldNumber` | Field number for the "opset_import" field. |
| `ProducerNameFieldNumber` | Field number for the "producer_name" field. |
| `ProducerVersionFieldNumber` | Field number for the "producer_version" field. |
| `TrainingInfoFieldNumber` | Field number for the "training_info" field. |

