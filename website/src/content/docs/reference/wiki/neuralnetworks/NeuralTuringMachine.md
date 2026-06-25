---
title: "NeuralTuringMachine<T>"
description: "Represents a Neural Turing Machine, which is a neural network architecture that combines a neural network with external memory."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Neural Turing Machine, which is a neural network architecture that combines a neural network with external memory.

## For Beginners

A Neural Turing Machine is like a neural network with a "notebook" that it can write to and read from.

Think of it like a student solving a math problem:

- The student (neural network) can process information directly
- But for complex problems, the student needs to write down intermediate steps in a notebook (external memory)
- The student can later refer back to these notes when needed

This memory capability helps the network:

- Remember information over long periods
- Store and retrieve specific pieces of data
- Learn more complex patterns that require step-by-step reasoning

For example, a standard neural network might struggle to add two long numbers, but an NTM can learn to write down 
partial results and carry digits, similar to how humans solve addition problems.

## How It Works

A Neural Turing Machine (NTM) extends traditional neural networks by adding an external memory component that
the network can read from and write to. This allows the network to store and retrieve information over long
sequences, making it particularly effective for tasks requiring complex memory operations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralTuringMachine` | Initializes a new instance of the `NeuralTuringMachine` class. |
| `NeuralTuringMachine(NeuralNetworkArchitecture<>,Int32,Int32,Int32,ILossFunction<>,IVectorActivationFunction<>,IVectorActivationFunction<>,IVectorActivationFunction<>,NeuralTuringMachineOptions)` | Initializes a new instance of the `NeuralTuringMachine` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for the memory usage auxiliary loss. |
| `ContentAddressingActivation` | The activation function to apply to content-based addressing similarity scores. |
| `ContentAddressingVectorActivation` | The activation function to apply to content-based addressing similarity scores. |
| `GateActivation` | The activation function to apply to interpolation gates. |
| `GateVectorActivation` | The activation function to apply to interpolation gates. |
| `OutputActivation` | The activation function to apply to the final output. |
| `OutputVectorActivation` | The activation function to apply to the final output. |
| `UseAuxiliaryLoss` | Gets or sets whether auxiliary loss (memory usage regularization) should be used during training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyActivation(Vector<>,NeuralTuringMachine<>.ActivationType)` | Applies the appropriate activation function to a vector. |
| `ApplyScalarActivation(Vector<>,IActivationFunction<>)` | Applies a scalar activation function element-wise to a vector. |
| `CombineSequenceOutputs(List<Tensor<>>)` | Combines sequence outputs into a single tensor. |
| `ComputeAttentionWeights(Vector<>,Vector<>,Matrix<>)` | Computes attention weights using content-based and location-based addressing. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for memory usage regularization. |
| `ContentAddressing(Matrix<>,Vector<>,)` | Applies content-based addressing to find similar memory locations. |
| `ConvolutionalShift(Vector<>,Vector<>)` | Applies a circular convolution to shift attention weights. |
| `CreateNewInstance` | Creates a new instance of the neural turing machine model. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes NTM-specific data from a binary reader. |
| `ExtractTimeStepInput(Tensor<>,Int32,Int32)` | Extracts input for a specific time step from the input tensor. |
| `ExtractTimeStepTensor(Tensor<>,Int32,Int32,Int32)` | Extracts the time-step `t` slice from `shapedInput`, preserving tape provenance via engine ops (Reshape only — no manual element copies). |
| `ExtractVector(Tensor<>,Int32)` | Extracts a vector from a tensor for a specific batch element. |
| `ForwardForTraining(Tensor<>)` | Forward path used by the training tape — routes directly through the NTM-specific tape-aware `Tensor{` memory pipeline (controller → read/write addressing → output), the SAME function `Predict` evaluates. |
| `ForwardTape(Tensor<>)` | Tape-aware forward pipeline. |
| `GenerateOutput(Tensor<>,Tensor<>)` | Generates the final output from controller state and read result. |
| `GenerateReadParameters(Tensor<>)` | Generates parameters for memory reading from controller output. |
| `GenerateWriteParameters(Tensor<>)` | Generates parameters for memory writing from controller output. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the memory usage auxiliary loss. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the Neural Turing Machine model. |
| `GetOptions` |  |
| `InitializeDefaultMemoryAndWeights` | Initializes default memory and attention weights. |
| `InitializeLayers` | Initializes the neural network layers based on the provided architecture. |
| `InitializeMemory` | Initializes the memory matrices with small random values and snapshots the result as the initial-state template used to reset working memory at the start of every forward pass. |
| `NextGaussian(Random,Double,Double)` | Draws a sample from Normal(mean, stdDev) using the Box-Muller transform on a caller-supplied (seedable) `Random`, so the initial-memory draw in `InitializeMemory` is reproducible under a fixed seed (#1670). |
| `PredictCore(Tensor<>)` | Performs a forward pass through the Neural Turing Machine. |
| `ProcessController(Tensor<>)` | Processes input through the controller network. |
| `ReadFromMemories` | Reads from all batch memories using their respective attention weights. |
| `ReadFromMemory(Matrix<>,Vector<>)` | Reads from memory using attention weights. |
| `ResetRuntimeState(Int32)` | Resets each batch element's working memory and read/write attention weights to their canonical initial state. |
| `ResetState` | Resets the internal state of the neural network. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes NTM-specific data to a binary writer. |
| `SetTrainingMode(Boolean)` | Sets the layer to training or evaluation mode. |
| `SetupBatchMemories(Int32)` | Sets up memories and attention weights for the given batch size. |
| `Sharpen(Vector<>,)` | Sharpens a weight vector by raising to a power and renormalizing. |
| `SliceColumnAsBx1(Tensor<>,Int32)` | Slices a single "column" (axis-1 index) from a rank-2 parameter tensor and returns it as a [B, 1] tensor. |
| `TapeClampNonNegative(Tensor<>)` | Clamp every element to ≥ 0 via `(x + \|x\|) / 2`, the standard piecewise-linear ReLU identity. |
| `TapeComputeAttention(Tensor<>,Tensor<>,Tensor<>)` | Paper-faithful attention pipeline: content addressing (§3.3.1) → interpolation (§3.3.2) → convolutional shift (§3.3.3) → sharpening (§3.3.4). |
| `TapeContentAddressing(Tensor<>,Tensor<>,Int32,Tensor<>)` | Content-based addressing (NTM §3.3.1): K(k, m) = (k · m) / (\|\|k\|\| · \|\|m\|\|) w_c = softmax(β · K) |
| `TapeConvolutionalShift(Tensor<>,Tensor<>)` | NTM §3.3.3 circular convolutional shift: `w_~(i) = Σ_j w_g((i-j) mod M) · s(j)`. |
| `TapeInterpolate(Tensor<>,Tensor<>,Tensor<>)` | NTM §3.3.2 interpolation: `w_g = g · w_c + (1-g) · w_prev`. |
| `TapeQuartile(Tensor<>,Int32,Int32)` | Extracts a column-quartile from `x` shape [B, W] using reshape + slice so the tape sees the operation. |
| `TapeReadFromMemory(Tensor<>,Tensor<>)` | Reads from memory by attention-weighted sum. |
| `TapeRoll(Tensor<>,Int32)` | Cyclic roll along axis 1 by `offset` positions. |
| `TapeSharpenTensor(Tensor<>,Tensor<>)` | NTM §3.3.4 sharpening, tensor-tape variant: `w(i) = w_~(i)^γ / Σ_j w_~(j)^γ`. |
| `TapeWriteHeads(Tensor<>)` | Splits the write parameters into erase ∈ [0,1] (sigmoid of the first half) and add ∈ ℝ (tanh of the second half) vectors. |
| `TapeWriteMemory(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | NTM §3.2 memory update: M_t(m, v) = M_{t-1}(m, v) · (1 - w(m) · e(v)) + w(m) · a(v) Implemented as broadcast tensor ops on shapes [B, M, V] / [B, M] / [B, V]. |
| `TileInitialMemory(Int32)` | Tiles `_initialMemoryTensor` across the batch dim, producing the per-batch working-memory tensor used as the starting point of each forward sequence. |
| `Train(Tensor<>,Tensor<>)` | Trains the Neural Turing Machine on a single batch of input-output pairs. |
| `UniformAttention(Int32)` | Returns a uniform [B, M] attention tensor with each entry = 1/M. |
| `UpdateAttentionWeights(Tensor<>,Tensor<>)` | Updates attention weights for both reading and writing operations. |
| `UpdateParameters()` | Updates the parameters of the neural network layers. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the neural network layers. |
| `WriteToMemories(Tensor<>)` | Writes to all batch memories using their respective attention weights. |
| `WriteToMemory(Matrix<>,Vector<>,Vector<>,Vector<>)` | Writes to memory using attention weights and erase/add vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_controllerSize` | The size of the controller network that manages memory operations. |
| `_initialMemoryTemplate` | Snapshot of the initial memory matrix taken at construction time. |
| `_initialMemoryTensor` | Tensor mirror of `_initialMemoryTemplate` with shape `[memorySize, memoryVectorSize]`. |
| `_isTraining` | Indicates whether the network is in training mode. |
| `_memories` | The external memory matrices used by the Neural Turing Machine, one per batch element. |
| `_memorySize` | The size of the external memory matrix (number of memory locations). |
| `_memoryVectorSize` | The size of each memory vector (the amount of information stored at each memory location). |
| `_readWeights` | The current reading weights for each batch element. |
| `_writeWeights` | The current writing weights for each batch element. |

