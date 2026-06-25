---
title: "GraphProto"
description: "Graphs  A graph defines the computational logic of a model and is comprised of a parameterized list of nodes that form a directed acyclic graph based on their inputs and outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Graphs

A graph defines the computational logic of a model and is comprised of a parameterized
list of nodes that form a directed acyclic graph based on their inputs and outputs.
This is the equivalent of the "network" or "graph" in many deep learning
frameworks.

## Properties

| Property | Summary |
|:-----|:--------|
| `DocString` | A human-readable documentation for this graph. |
| `Initializer` | A list of named tensor values, used to specify constant inputs of the graph. |
| `Input` | The inputs and outputs of the graph. |
| `MetadataProps` | Named metadata values; keys should be distinct. |
| `Name` | The name of the graph. |
| `Node` | The nodes in the graph, sorted topologically. |
| `QuantizationAnnotation` | This field carries information to indicate the mapping among a tensor and its quantization parameter tensors. |
| `SparseInitializer` | Initializers (see above) stored in sparse format. |
| `ValueInfo` | Information for the values in the graph. |

## Fields

| Field | Summary |
|:-----|:--------|
| `DocStringFieldNumber` | Field number for the "doc_string" field. |
| `InitializerFieldNumber` | Field number for the "initializer" field. |
| `InputFieldNumber` | Field number for the "input" field. |
| `MetadataPropsFieldNumber` | Field number for the "metadata_props" field. |
| `NameFieldNumber` | Field number for the "name" field. |
| `NodeFieldNumber` | Field number for the "node" field. |
| `OutputFieldNumber` | Field number for the "output" field. |
| `QuantizationAnnotationFieldNumber` | Field number for the "quantization_annotation" field. |
| `SparseInitializerFieldNumber` | Field number for the "sparse_initializer" field. |
| `ValueInfoFieldNumber` | Field number for the "value_info" field. |

