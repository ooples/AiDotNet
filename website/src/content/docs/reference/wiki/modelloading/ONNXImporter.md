---
title: "ONNXImporter<T>"
description: "Imports weights from ONNX model files."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelLoading`

Imports weights from ONNX model files.

## For Beginners

ONNX (Open Neural Network Exchange) is a standard format
for representing machine learning models.

Many pretrained models are distributed in ONNX format. This class extracts
the learned weights from ONNX files so you can use them in your models.

Example usage:
```cs
var importer = new ONNXImporter<float>();

// Load weights from ONNX file
var weights = importer.LoadWeights("model.onnx");

// Apply to your model
var layer = new DenseLayer<float>(inputSize, outputSize);
importer.ApplyWeights(layer, weights);
```

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ONNXImporter(Boolean)` | Initializes a new instance of the ONNXImporter class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyWeights(IWeightLoadable<>,Dictionary<String,Tensor<>>,Func<String,String>,Boolean)` | Applies loaded weights to a model using IWeightLoadable. |
| `ConvertToTensor(ONNXImporter<>.ONNXInitializer)` | Converts an ONNX initializer to a Tensor. |
| `GetDataTypeName(Int32)` | Gets the human-readable name for an ONNX data type. |
| `GetTensorInfo(String)` | Gets information about tensors in an ONNX file without loading them. |
| `LoadWeights(String)` | Loads all initializer tensors from an ONNX file. |
| `ParseGraphForInitializers(BinaryReader,Int64,List<ONNXImporter<>.ONNXInitializer>,Boolean)` | Parses the graph section for initializer tensors. |
| `ParseONNXInitializers(BinaryReader,Int64,Boolean)` | Parses ONNX file to extract initializer tensors. |
| `ParseTensorProto(BinaryReader,Int64,Boolean)` | Parses a TensorProto message. |
| `ReadVarint(BinaryReader)` | Reads a protobuf varint. |
| `SkipField(BinaryReader,Int32)` | Skips a protobuf field based on wire type. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Provides numeric operations for the specific type T. |
| `_verbose` | Whether to log import progress. |

