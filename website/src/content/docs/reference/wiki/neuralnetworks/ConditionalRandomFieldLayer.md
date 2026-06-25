---
title: "ConditionalRandomFieldLayer<T>"
description: "Represents a Conditional Random Field (CRF) layer for sequence labeling tasks."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents a Conditional Random Field (CRF) layer for sequence labeling tasks.

## For Beginners

A Conditional Random Field (CRF) layer is used when you need to label each item 
in a sequence while considering how labels relate to each other.

In many sequence tasks, the label for an item depends not just on the item itself, but also on nearby items:

For example, in a sentence like "John Smith lives in New York":

- Without CRF: Each word might be labeled independently, potentially creating invalid sequences
- With CRF: The model considers that "New" followed by "York" is likely a location name

Think of it like:

- Standard layers ask, "What's the best label for this word on its own?"
- CRF layers ask, "What's the best sequence of labels for the whole sentence?"

CRFs are especially useful for tasks like:

- Named entity recognition (finding names of people, organizations, locations)
- Part-of-speech tagging (labeling words as nouns, verbs, etc.)
- Any task where the correct labels form patterns or follow rules

## How It Works

A Conditional Random Field (CRF) layer is a specialized neural network layer designed for sequence labeling
tasks such as named entity recognition, part-of-speech tagging, and activity recognition. Unlike standard
classification layers that make independent predictions for each element in a sequence, CRF layers model
the dependencies between labels in a sequence, leading to more coherent predictions. The layer uses the
Viterbi algorithm to find the most likely sequence of labels given the input features and learned transition
probabilities between labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConditionalRandomFieldLayer(Int32,IActivationFunction<>)` | Lazy constructor: resolves `sequenceLength` from `input.Shape[0]` on first `Tensor{`. |
| `ConditionalRandomFieldLayer(Int32,IVectorActivationFunction<>)` | Lazy constructor with vector activation — resolves `sequenceLength` from `input.Shape[0]` on first `Tensor{`. |
| `ConditionalRandomFieldLayer(Int32,Int32,IActivationFunction<>)` | Initializes a new instance of the `ConditionalRandomFieldLayer` class with a scalar activation function. |
| `ConditionalRandomFieldLayer(Int32,Int32,IVectorActivationFunction<>)` | Initializes a new instance of the `ConditionalRandomFieldLayer` class with a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets a value indicating whether this layer supports training. |
| `SupportsGpuExecution` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildLabelOneHotForBatch(Tensor<>,Int32,Int32,Int32)` | Builds a [seqLen, numClasses] one-hot encoding of the gold labels for a single batch element. |
| `ComputeNegativeLogLikelihood(Tensor<>,Tensor<>)` | Computes the linear-chain CRF negative log-likelihood for a batched emissions tensor and the corresponding gold-label sequence: `NLL(emissions, labels) = logZ(emissions) − goldScore(emissions, labels)` where `logZ` is the partition function… |
| `EnsureInitialized` |  |
| `Forward(Tensor<>)` | Performs the forward pass of the CRF layer. |
| `ForwardGpu(Tensor<>[])` |  |
| `GetMetadata` | Persists CRF-specific constructor parameters so deserialization can reconstruct the layer with the same `numClasses` and `sequenceLength`. |
| `GetParameterGradients` | Sets the trainable parameters for the layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters from the layer as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeParameters` | Initializes the layer's parameters with scaled random values. |
| `OnFirstForward(Tensor<>)` |  |
| `ResetState` | Resets the internal state of the CRF layer. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SliceLabelOneHotSubrange(Tensor<>,Int32,Int32)` | Returns a tape-tracked slice of a [seqLen, numClasses] one-hot tensor covering rows `[start, start+count)`. |
| `TapeLogSumExpAxis(Tensor<>,Int32)` | Tape-tracked log-sum-exp along a single axis, returning the reduced tensor with that axis removed. |
| `UpdateParameters()` | Updates the layer's parameters using the calculated gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_originalInputShape` | Stores the original input shape for any-rank tensor support. |

