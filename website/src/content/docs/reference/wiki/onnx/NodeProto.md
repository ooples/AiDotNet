---
title: "NodeProto"
description: "Nodes  Computation graphs are made up of a DAG of nodes, which represent what is commonly called a \"layer\" or \"pipeline stage\" in machine learning frameworks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Nodes

Computation graphs are made up of a DAG of nodes, which represent what is
commonly called a "layer" or "pipeline stage" in machine learning frameworks.

For example, it can be a node of type "Conv" that takes in an image, a filter
tensor and a bias tensor, and produces the convolved output.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attribute` | Additional named attributes. |
| `DocString` | A human-readable documentation for this node. |
| `Domain` | The domain of the OperatorSet that specifies the operator named by op_type. |
| `Input` | namespace Value |
| `MetadataProps` | Named metadata values; keys should be distinct. |
| `Name` | An optional identifier for this node in a graph. |
| `OpType` | The symbolic identifier of the Operator to execute. |
| `Output` | namespace Value |
| `Overload` | Overload identifier, used only to map this to a model-local function. |

## Fields

| Field | Summary |
|:-----|:--------|
| `AttributeFieldNumber` | Field number for the "attribute" field. |
| `DocStringFieldNumber` | Field number for the "doc_string" field. |
| `DomainFieldNumber` | Field number for the "domain" field. |
| `InputFieldNumber` | Field number for the "input" field. |
| `MetadataPropsFieldNumber` | Field number for the "metadata_props" field. |
| `NameFieldNumber` | Field number for the "name" field. |
| `OpTypeFieldNumber` | Field number for the "op_type" field. |
| `OutputFieldNumber` | Field number for the "output" field. |
| `OverloadFieldNumber` | Field number for the "overload" field. |

