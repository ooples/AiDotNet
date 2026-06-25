---
title: "DifferentiableNeuralComputer<T>"
description: "Represents a Differentiable Neural Computer (DNC), a neural network architecture that combines neural networks with external memory resources."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Differentiable Neural Computer (DNC), a neural network architecture that combines neural networks with external memory resources.

## For Beginners

A Differentiable Neural Computer is like a neural network with a notepad.

Imagine a traditional neural network as a person who can make decisions based on what they see,
but can only keep information in their head. A DNC is like giving that person a notepad to:

- Write down important information
- Organize notes in a systematic way
- Look back at previously written notes when making decisions
- Learn which information is worth writing down and when to refer back to it

This combination of neural processing with external memory allows the DNC to solve problems that
require remembering and reasoning about complex relationships or sequences of information, like
navigating a subway map or following a multi-step recipe.

## How It Works

A Differentiable Neural Computer (DNC) is an advanced neural network architecture that augments neural networks with
an external memory matrix and mechanisms to read from and write to this memory. DNCs can learn to use their memory
to store and retrieve information, enabling them to solve complex, structured problems that require reasoning and
algorithm-like behavior. The key components include a controller neural network, a memory matrix, and read/write heads
that interact with the memory through differentiable attention mechanisms.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DifferentiableNeuralComputer` | Initializes a new instance of the `DifferentiableNeuralComputer` class with the specified parameters. |
| `DifferentiableNeuralComputer(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,IVectorActivationFunction<>,DifferentiableNeuralComputerOptions)` | Initializes a new instance of the `DifferentiableNeuralComputer` class with the specified parameters. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the memory addressing auxiliary loss. |
| `SupportsTraining` |  |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (memory addressing regularization) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateDNCInterfaceSize(Int32,Int32)` | Calculates the size of the interface vector required for memory operations. |
| `ColumnToLeaf([])` | Builds a detached [N, 1] column tape leaf from an [N] array. |
| `CombineControllerOutputWithReadVectors(Tensor<>,Tensor<>)` | Combines the controller output with read vectors to produce the final output. |
| `ComputeAllocation([])` | Computes the dynamic memory allocation weighting from the usage vector (Graves et al. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for memory addressing regularization. |
| `ComputeReadVectorsDifferentiable(Tensor<>)` | Full Differentiable Neural Computer memory interaction (Graves et al. |
| `ContentAddressingTensor(Tensor<>,Tensor<>,Tensor<>)` | Tape-tracked cosine-similarity content addressing: softmax over locations of strength · cos(memoryRow, key). |
| `CreateNewInstance` | Creates a new instance of the differentiable neural computer model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes Differentiable Neural Computer-specific data from a binary reader. |
| `ForwardForTraining(Tensor<>)` |  |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the memory addressing auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the Differentiable Neural Computer model. |
| `GetOptions` |  |
| `InitializeLayers` | Initializes the layers of the Differentiable Neural Computer based on the architecture. |
| `InitializeMemory` | Initializes the memory matrix and usage tracking vector. |
| `LinkMatVec(Matrix<>,Vector<>,Boolean)` | Computes L·v (forward) or Lᵀ·v (backward) for the temporal link matrix, returning an [N] array. |
| `MemoryToLeaf` | Builds a detached [N, W] tape leaf from the carried memory matrix `_memory`. |
| `MirrorMemoryState(Tensor<>,[],[],Vector<>,Matrix<>,List<Vector<>>)` | Mirrors the step's evolving DNC state (memory, usage, write/read weightings, precedence, temporal links) back into the carried detached fields, so sequence processing, serialization, and diagnostics observe the real state. |
| `PinElements(Tensor<>)` | Processes an input through the DNC, updating memory state and producing an output. |
| `PredictCore(Tensor<>)` | Makes a prediction using the Differentiable Neural Computer. |
| `PrepareControllerInput(Tensor<>)` | Prepares the input for the controller by concatenating it with previous read vectors. |
| `ProcessSequence(List<Tensor<>>,Boolean)` | Processes a sequence of inputs through the DNC. |
| `ProcessThroughController(Tensor<>)` | Processes the prepared input through the controller network. |
| `ResetMemoryState` | Resets the state of the Differentiable Neural Computer. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes Differentiable Neural Computer-specific data to a binary writer. |
| `UpdateParameters(Vector<>)` | Updates the parameters of all layers in the Differentiable Neural Computer. |
| `UpdateReadVectorState(List<Tensor<>>)` | Mirrors the tape-tracked read vectors back into the detached `_readVectors` / `_memory` bookkeeping used by the next step's `Tensor{` (no gradient role). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_activationFunction` | Gets or sets the scalar activation function for the network. |
| `_controllerSize` | Gets or sets the size of the controller network's output. |
| `_memory` | Gets or sets the memory matrix that stores information. |
| `_memorySize` | Gets or sets the number of memory locations in the memory matrix. |
| `_memoryWordSize` | Gets or sets the size of each memory word or location in the memory matrix. |
| `_outputWeights` | The output weight matrix for combining controller output with read vectors. |
| `_precedenceWeighting` | Gets or sets the vector that tracks the order in which memory locations were written to. |
| `_readHeads` | Gets or sets the number of read heads that can access the memory simultaneously. |
| `_readVectors` | Gets or sets the list of vectors read from memory. |
| `_readWeightings` | Gets or sets the list of vectors that determine where to read from memory. |
| `_suppressMemoryReset` | When true, `Boolean)` does NOT reset the memory state at the start of the forward pass, so the carried memory/usage/link state persists across calls. |
| `_temporalLinkMatrix` | Gets or sets the matrix representing temporal links between memory locations. |
| `_trainOptimizer` | Trains the Differentiable Neural Computer on a single batch of data. |
| `_usageFree` | Gets or sets the vector tracking which memory locations are free to be written to. |
| `_vectorActivationFunction` | Gets or sets the vector activation function for the network. |
| `_writeWeighting` | Gets or sets the vector that determines where to write in memory. |

