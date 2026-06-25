---
title: "TrainingInfoProto"
description: "Training information TrainingInfoProto stores information for training a model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Onnx.Protobuf`

Training information
TrainingInfoProto stores information for training a model.
In particular, this defines two functionalities: an initialization-step
and a training-algorithm-step. Initialization resets the model
back to its original state as if no training has been performed.
Training algorithm improves the model based on input data.

The semantics of the initialization-step is that the initializers
in ModelProto.graph and in TrainingInfoProto.algorithm are first
initialized as specified by the initializers in the graph, and then
updated by the "initialization_binding" in every instance in
ModelProto.training_info.

The field "algorithm" defines a computation graph which represents a
training algorithm's step. After the execution of a
TrainingInfoProto.algorithm, the initializers specified by "update_binding"
may be immediately updated. If the targeted training algorithm contains
consecutive update steps (such as block coordinate descent methods),
the user needs to create a TrainingInfoProto for each step.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | This field represents a training algorithm step. |
| `Initialization` | This field describes a graph to compute the initial tensors upon starting the training process. |
| `InitializationBinding` | This field specifies the bindings from the outputs of "initialization" to some initializers in "ModelProto.graph.initializer" and the "algorithm.initializer" in the same TrainingInfoProto. |
| `UpdateBinding` | Gradient-based training is usually an iterative procedure. |

## Fields

| Field | Summary |
|:-----|:--------|
| `AlgorithmFieldNumber` | Field number for the "algorithm" field. |
| `InitializationBindingFieldNumber` | Field number for the "initialization_binding" field. |
| `InitializationFieldNumber` | Field number for the "initialization" field. |
| `UpdateBindingFieldNumber` | Field number for the "update_binding" field. |

