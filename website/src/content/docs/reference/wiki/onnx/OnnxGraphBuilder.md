---
title: "OnnxGraphBuilder"
description: "Thin facade over the vendored ONNX protobuf types that lets layer converters add nodes, initializers, inputs, and outputs to a model graph without touching the generated `GraphProto` directly."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Onnx`

Thin facade over the vendored ONNX protobuf types that lets layer converters add
nodes, initializers, inputs, and outputs to a model graph without touching the
generated `GraphProto` directly.

One `OnnxGraphBuilder` is created per export. Layer converters call
`NodeProto)`, `TensorProto)`, `ValueInfoProto)`,
`ValueInfoProto)`, and `String)` as needed; the final
`Build` assembles a `ModelProto` ready to write.

This class is intentionally minimal in this initial commit — layer converters
landing in subsequent commits drive the API additions they need.

## Properties

| Property | Summary |
|:-----|:--------|
| `OpsetVersion` | Opset version the resulting model targets. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddFloatInitializer(String,Single[],Int32[])` | Adds a float32 constant initializer with the given shape. |
| `AddFloatInput(String,Int32[])` | Declares a float32 graph input with the given (fixed or symbolic) shape. |
| `AddFloatOutput(String,Int32[])` | Declares a float32 graph output. |
| `AddInitializer(TensorProto)` | Adds a constant initializer (e.g., a layer's weight tensor). |
| `AddInput(ValueInfoProto)` | Adds a graph input (a tensor the model expects from the caller). |
| `AddNode(NodeProto)` | Adds a node to the graph. |
| `AddOp(String,String[],String[],String)` | Adds an ONNX op node with named inputs and outputs and optional attributes. |
| `AddOutput(ValueInfoProto)` | Adds a graph output (a tensor the model produces). |
| `Build` | Assembles the final ModelProto with producer metadata + opset declaration. |
| `NextTensorName(String)` | Reserves a unique tensor name with the given prefix. |
| `WriteTo(Stream)` | Serializes the built ModelProto to a stream. |

