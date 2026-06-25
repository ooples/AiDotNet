---
title: "HopeNetwork<T>"
description: "Hope architecture - a self-modifying recurrent neural network variant of Titans with unbounded levels of in-context learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Hope architecture - a self-modifying recurrent neural network variant of Titans
with unbounded levels of in-context learning.
Core innovation of Google's Nested Learning paradigm.

## For Beginners

HopeNetwork is a self-modifying neural network inspired by
Google's Nested Learning paradigm. Unlike standard networks with fixed architectures, it
can modify its own behavior during inference through a continuum memory system. This allows
it to perform unbounded levels of in-context learning, meaning it can keep adapting to new
patterns without being retrained. Think of it as a network that can "learn to learn" in
real time.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HopeNetwork` | Initializes a new instance with default architecture settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationStep` | Gets the adaptation step count. |
| `InContextLearningLevels` | Gets the number of in-context learning levels (unbounded in theory, bounded in practice). |
| `SupportsTraining` | Indicates whether the network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddOutputLayer(Int32,ActivationFunction)` | Adds an output layer to the Hope network. |
| `Clone` | Cloning HopeNetwork via the default DeepCopy path (serialize/deserialize) produces a network whose Predict output drifts from the original by roughly 1e-7 even though every parameter and meta-state value matches bit-exactly. |
| `ConsolidateMemory` | Consolidates memories across all CMS blocks. |
| `CreateNewInstance` | Creates a new instance of HopeNetwork with the same architecture. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Hope-specific data for model restoration. |
| `Forward(Tensor<>)` | Performs a forward pass through the Hope architecture. |
| `GetAssociativeMemory` | Gets the associative memory system. |
| `GetCMSBlocks` | Gets the CMS blocks (for inspection/debugging). |
| `GetContextFlow` | Gets the context flow mechanism. |
| `GetMetaState` | Gets the current meta-state (for inspection/debugging). |
| `GetModelMetadata` | Gets metadata about the model (required by NeuralNetworkBase). |
| `GetOptions` |  |
| `IsMetaStateZero(Vector<>)` | Applies self-modification to input based on meta-state. |
| `PredictCore(Tensor<>)` | Makes a prediction on the given input (required by NeuralNetworkBase). |
| `ResetMemory` | Resets all memory in CMS blocks and meta-state. |
| `ResetRecurrentState` | Resets recurrent layer states. |
| `ResetState` | Resets the state of the network (required by NeuralNetworkBase). |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Hope-specific data for model persistence. |
| `SetSelfModificationRate()` | Sets the self-modification rate for self-referential optimization. |
| `UpdateMetaStateSelfReferential(Tensor<>)` | Updates meta-state through self-referential optimization. |
| `UpdateParameters(Vector<>)` | Updates all parameters in the network (required by NeuralNetworkBase). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_trainOptimizer` | Trains the network on a single input-output pair (required by NeuralNetworkBase). |

