---
title: "ONNX"
description: "All 55 public types in the AiDotNet.onnx namespace, organized by kind."
section: "API Reference"
---

**55** public types in this namespace, organized by kind.

## Models & Types (27)

| Type | Summary |
|:-----|:--------|
| [`AttributeProto`](/docs/reference/wiki/onnx/attributeproto/) | Attributes  A named attribute containing either singular float, integer, string, graph, and tensor values, or repeated float, integer, string, graph, and tensor values. |
| [`Dimension`](/docs/reference/wiki/onnx/dimension/) |  |
| [`FunctionProto`](/docs/reference/wiki/onnx/functionproto/) |  |
| [`GraphProto`](/docs/reference/wiki/onnx/graphproto/) | Graphs  A graph defines the computational logic of a model and is comprised of a parameterized list of nodes that form a directed acyclic graph based on their inputs and outputs. |
| [`Map`](/docs/reference/wiki/onnx/map/) | map<K,V> |
| [`ModelProto`](/docs/reference/wiki/onnx/modelproto/) | Models  ModelProto is a top-level file/container format for bundling a ML model and associating its computation graph with metadata. |
| [`NodeProto`](/docs/reference/wiki/onnx/nodeproto/) | Nodes  Computation graphs are made up of a DAG of nodes, which represent what is commonly called a "layer" or "pipeline stage" in machine learning frameworks. |
| [`OnnxLayerInputs`](/docs/reference/wiki/onnx/onnxlayerinputs/) | Named tensors flowing INTO a layer's ONNX node(s). |
| [`OnnxLayerOutputs`](/docs/reference/wiki/onnx/onnxlayeroutputs/) | Named tensors flowing OUT of a layer's ONNX node(s). |
| [`OnnxModelDownloader`](/docs/reference/wiki/onnx/onnxmodeldownloader/) | Downloads ONNX models from HuggingFace Hub and other repositories. |
| [`OnnxModelMetadata`](/docs/reference/wiki/onnx/onnxmodelmetadata/) | Metadata about a loaded ONNX model. |
| [`OnnxModel<T>`](/docs/reference/wiki/onnx/onnxmodel/) | A wrapper for ONNX models that provides easy-to-use inference with AiDotNet Tensor types. |
| [`OnnxTensorInfo`](/docs/reference/wiki/onnx/onnxtensorinfo/) | Information about an ONNX tensor (input or output). |
| [`OperatorSetIdProto`](/docs/reference/wiki/onnx/operatorsetidproto/) | Operator Sets  OperatorSets are uniquely identified by a (domain, opset_version) pair. |
| [`Optional`](/docs/reference/wiki/onnx/optional/) | wrapper for Tensor, Sequence, or Map |
| [`Segment`](/docs/reference/wiki/onnx/segment/) | For very large tensors, we may want to store them in chunks, in which case the following fields will specify the segment that is stored in the current TensorProto. |
| [`Sequence`](/docs/reference/wiki/onnx/sequence/) | repeated T |
| [`SparseTensor`](/docs/reference/wiki/onnx/sparsetensor/) |  |
| [`SparseTensorProto`](/docs/reference/wiki/onnx/sparsetensorproto/) | A serialized sparse-tensor value |
| [`StringStringEntryProto`](/docs/reference/wiki/onnx/stringstringentryproto/) | StringStringEntryProto follows the pattern for cross-proto-version maps. |
| [`Tensor`](/docs/reference/wiki/onnx/tensor/) |  |
| [`TensorAnnotation`](/docs/reference/wiki/onnx/tensorannotation/) |  |
| [`TensorProto`](/docs/reference/wiki/onnx/tensorproto/) | Tensors  A serialized tensor value. |
| [`TensorShapeProto`](/docs/reference/wiki/onnx/tensorshapeproto/) | Defines a tensor shape. |
| [`TrainingInfoProto`](/docs/reference/wiki/onnx/traininginfoproto/) | Training information TrainingInfoProto stores information for training a model. |
| [`TypeProto`](/docs/reference/wiki/onnx/typeproto/) | Types  The standard ONNX data types. |
| [`ValueInfoProto`](/docs/reference/wiki/onnx/valueinfoproto/) | Defines information on value, including the name, the type, and the shape of the value. |

## Enums (10)

| Type | Summary |
|:-----|:--------|
| [`AttributeType`](/docs/reference/wiki/onnx/attributetype/) | Note: this enum is structurally identical to the OpSchema::AttrType enum defined in schema.h. |
| [`DataLocation`](/docs/reference/wiki/onnx/datalocation/) | Location of the data for this tensor. |
| [`DataType`](/docs/reference/wiki/onnx/datatype/) |  |
| [`GraphOptimizationLevel`](/docs/reference/wiki/onnx/graphoptimizationlevel/) | Graph optimization levels for ONNX Runtime. |
| [`OnnxExecutionProvider`](/docs/reference/wiki/onnx/onnxexecutionprovider/) | Specifies the execution provider (hardware accelerator) for ONNX model inference. |
| [`OnnxLogLevel`](/docs/reference/wiki/onnx/onnxloglevel/) | Log severity levels for ONNX Runtime. |
| [`OperatorStatus`](/docs/reference/wiki/onnx/operatorstatus/) | Operator/function status. |
| [`ValueOneofCase`](/docs/reference/wiki/onnx/valueoneofcase/) | Enum of possible cases for the "value" oneof. |
| [`ValueOneofCase`](/docs/reference/wiki/onnx/valueoneofcase-2/) | Enum of possible cases for the "value" oneof. |
| [`Version`](/docs/reference/wiki/onnx/version/) | Versioning  ONNX versioning is specified in docs/IR.md and elaborated on in docs/Versioning.md  To be compatible with both proto2 and proto3, we will use a version number that is not defined by the default value but an explicit enum number. |

## Structs (1)

| Type | Summary |
|:-----|:--------|
| [`OnnxAxisSpec`](/docs/reference/wiki/onnx/onnxaxisspec/) | Per-axis shape descriptor for an ONNX `TensorShapeProto.Dimension`. |

## Options & Configuration (2)

| Type | Summary |
|:-----|:--------|
| [`OnnxExportOptions`](/docs/reference/wiki/onnx/onnxexportoptions/) | Optional configuration for ONNX export. |
| [`OnnxModelOptions`](/docs/reference/wiki/onnx/onnxmodeloptions/) | Configuration options for loading and running ONNX models. |

## Helpers & Utilities (14)

| Type | Summary |
|:-----|:--------|
| [`AiModelResultOnnxExtensions`](/docs/reference/wiki/onnx/aimodelresultonnxextensions/) | Public-facing ONNX export API on `AiModelResult`. |
| [`AudioGen`](/docs/reference/wiki/onnx/audiogen/) | Audio generation models. |
| [`OnnxExporter`](/docs/reference/wiki/onnx/onnxexporter/) | Exports AiDotNet models to the ONNX format. |
| [`OnnxGraphBuilder`](/docs/reference/wiki/onnx/onnxgraphbuilder/) | Thin facade over the vendored ONNX protobuf types that lets layer converters add nodes, initializers, inputs, and outputs to a model graph without touching the generated `GraphProto` directly. |
| [`OnnxModelBuilder`](/docs/reference/wiki/onnx/onnxmodelbuilder/) | Internal builder for constructing ONNX model bytes. |
| [`OnnxModelRepositories`](/docs/reference/wiki/onnx/onnxmodelrepositories/) | Common ONNX model repositories and their identifiers. |
| [`OnnxReflection`](/docs/reference/wiki/onnx/onnxreflection/) | Holder for reflection information generated from onnx.proto3 |
| [`OnnxTensorConverter`](/docs/reference/wiki/onnx/onnxtensorconverter/) | Converts between AiDotNet Tensor types and ONNX Runtime tensor types. |
| [`Tts`](/docs/reference/wiki/onnx/tts/) | Text-to-speech models. |
| [`Types`](/docs/reference/wiki/onnx/types/) | Container for nested types declared in the AttributeProto message type. |
| [`Types`](/docs/reference/wiki/onnx/types-2/) | Container for nested types declared in the TensorProto message type. |
| [`Types`](/docs/reference/wiki/onnx/types-3/) | Container for nested types declared in the TensorShapeProto message type. |
| [`Types`](/docs/reference/wiki/onnx/types-4/) | Container for nested types declared in the TypeProto message type. |
| [`Whisper`](/docs/reference/wiki/onnx/whisper/) | OpenAI Whisper speech recognition models. |

## Exceptions (1)

| Type | Summary |
|:-----|:--------|
| [`OnnxExportUnsupportedException`](/docs/reference/wiki/onnx/onnxexportunsupportedexception/) | Thrown when a model contains a layer or component that does not yet have an ONNX export converter. |

