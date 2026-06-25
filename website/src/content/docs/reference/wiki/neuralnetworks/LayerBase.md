---
title: "LayerBase<T>"
description: "Represents the base class for all neural network layers, providing common functionality and interfaces."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Represents the base class for all neural network layers, providing common functionality and interfaces.

## For Beginners

This is the blueprint that all neural network layers follow.

Think of LayerBase as the common foundation that all layers are built upon:

- It defines what every layer must be able to do (process data forward and backward)
- It provides shared tools that all layers can use (like activation functions)
- It manages the shapes of data flowing in and out of layers
- It handles saving and loading layer parameters

All specific layer types (like convolutional, dense, etc.) inherit from this class,
which ensures they all work together consistently in a neural network.

## How It Works

LayerBase is an abstract class that serves as the foundation for all neural network layers. It defines 
the common structure and functionality that all layers must implement, such as forward and backward 
propagation, parameter management, and activation functions. This class handles the core mechanics 
of layers in a neural network, allowing derived classes to focus on their specific implementations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LayerBase(Int32[],Int32[])` | Initializes a new instance of the `LayerBase` class with the specified input and output shapes. |
| `LayerBase(Int32[],Int32[],IActivationFunction<>)` | Initializes a new instance of the `LayerBase` class with the specified shapes and element-wise activation function. |
| `LayerBase(Int32[],Int32[],IVectorActivationFunction<>)` | Initializes a new instance of the `LayerBase` class with the specified shapes and vector activation function. |
| `LayerBase(Int32[][],Int32[])` | Initializes a new instance of the `LayerBase` class with multiple input shapes and a specified output shape. |
| `LayerBase(Int32[][],Int32[],IActivationFunction<>)` | Initializes a new instance of the `LayerBase` class with multiple input shapes, a specified output shape, and an element-wise activation function. |
| `LayerBase(Int32[][],Int32[],IVectorActivationFunction<>)` | Initializes a new instance of the `LayerBase` class with multiple input shapes, a specified output shape, and a vector activation function. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CanExecuteOnGpu` | Gets whether this layer can execute its forward pass on GPU. |
| `CanTrainOnGpu` | Gets whether this layer can execute GPU training (forward, backward, parameter update). |
| `CurrentPrecision` | Gets the precision type this layer should use based on the current mixed-precision policy. |
| `Engine` | Gets the global execution engine for vector operations. |
| `InitializationStrategy` | Gets or sets the initialization strategy for this layer. |
| `InputShape` | Gets the input shape for this layer. |
| `InputShapes` | Gets the input shapes for this layer, supporting multiple inputs. |
| `IsInferringShapes` | Whether the current thread is running a shape-only resolution forward. |
| `IsInitialized` | Gets a value indicating whether this layer has been initialized. |
| `IsMixedPrecisionActive` | Gets whether mixed-precision training is currently active. |
| `IsResolvingShapesOnly` | True only while `Int32[])` is on the stack. |
| `IsShapeResolved` |  |
| `KeepActivationCacheUnderTape` | #1624 escape hatch / test hook — see `ShouldCacheActivationsForManualBackward`. |
| `LayerName` | Gets the name of this layer for mixed-precision policy lookup. |
| `LowPrecisionResident` | Inference-only: keep this layer's large weight matrices RESIDENT at half precision (fp16) and upcast to the compute type transiently per forward. |
| `NamedParameterCount` | Gets the total number of named parameters. |
| `NumOps` | Gets the numeric operations provider for type T. |
| `OutputShape` | Gets the output shape for this layer. |
| `Random` | Gets the thread-safe random number generator. |
| `RandomSeed` | Per-layer deterministic random seed. |
| `ScalarActivation` | Gets the element-wise activation function for this layer, if specified. |
| `ShouldCacheActivationsForManualBackward` | True when the layer should populate its manual-backward activation caches: only when NO `GradientTape` is recording (the tape-less manual-autodiff path may read them) or when `KeepActivationCacheUnderTape` forces it. |
| `ShouldUseFP32` | Gets whether this layer should use full precision (FP32) even during mixed-precision training. |
| `SupportsGpuExecution` | Gets whether this layer has a GPU execution implementation for inference. |
| `SupportsGpuTraining` | Gets whether this layer has full GPU training support (forward, backward, and parameter updates). |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |
| `UseAutodiff` | Gets or sets a value indicating whether this layer uses automatic differentiation for backward passes. |
| `UseStreamingAllocator` | Routes lazy-layer weight allocation through the streaming pool's reservation-then-allocate path when this layer's parent network has engaged weight streaming, falling back to a plain `new Tensor<T>(shape)` otherwise. |
| `UsingVectorActivation` | Gets a value indicating whether this layer uses a vector activation function. |
| `VectorActivation` | Gets the vector activation function for this layer, if specified. |
| `Workspace` | Per-layer workspace for zero-allocation forward passes. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ActivateTensor(IActivationFunction<>,Tensor<>)` | Applies a scalar activation function to each element of a tensor. |
| `ActivateTensor(IVectorActivationFunction<>,Tensor<>)` | Applies a vector activation function to a tensor. |
| `AllSubLayersShapeResolved` | True iff every registered sub-layer has finished its lazy-shape resolution (and thus has a stable trainable-parameter set). |
| `AllocateLazyWeight(Int32[],Func<Tensor<>>)` | Allocates a lazy-init weight tensor of the given shape, routing through `Int32[])` when `UseStreamingAllocator` is true (so the pool can pre-evict competing weights to disk before this allocation hits the GC heap), or via the caller's `nonS… |
| `AppendTrainableParameter(Tensor<>,PersistentTensorRole)` | Appends a trainable parameter without role-based deduplication. |
| `ApplyActivation(Tensor<>)` | Applies the activation function to a tensor and caches the pre-activation input for correct derivative computation in the backward pass. |
| `ApplyActivation(Vector<>)` | Applies the activation function to a vector. |
| `ApplyActivationDerivative(,)` | Applies the derivative of the activation function to a single value. |
| `ApplyActivationDerivative(Tensor<>,Tensor<>)` | Applies the derivative of the activation function to a tensor. |
| `ApplyActivationDerivative(Vector<>,Vector<>)` | Applies the derivative of the activation function to a vector. |
| `ApplyActivationDerivativeFromOutput(Tensor<>,Tensor<>)` | Applies activation derivative given the POST-activation output value. |
| `ApplyActivationForwardGpu(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32)` | Applies the layer's activation function forward pass on GPU using the activation's own GPU method. |
| `ApplyGpuActivation(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,Int32,FusedActivationType)` | Applies the specified activation function on GPU using the direct backend operations. |
| `ApplyGpuActivationBackward(IDirectGpuBackend,IGpuBuffer,IGpuBuffer,IGpuBuffer,IGpuBuffer,Int32,FusedActivationType,Single)` | Applies the backward pass of the specified activation function on GPU. |
| `AssignInitializationSeedFromScope` | Assigns `RandomSeed` from the active `LayerInitializationSeedScope` when no seed has been set yet. |
| `CalculateInputShape(Int32,Int32,Int32)` | Calculates a standard input shape for 2D data with batch size of 1. |
| `CalculateOutputShape(Int32,Int32,Int32)` | Calculates a standard output shape for 2D data with batch size of 1. |
| `CaptureScalarActivationParameters(Dictionary<String,String>,Object)` | Inspects a scalar activation for well-known parametric properties (currently `Alpha` covering LeakyReLU / ELU / PReLU / RReLU / SELU) and stores them on the layer metadata so `DeserializationHelper`'s `TryCreateActivationInstance` can rebui… |
| `ClearGradients` | Clears all parameter gradients in this layer. |
| `ClearRegisteredParameters` | Clears the registered trainable parameter list. |
| `Clone` | Creates a copy of this layer. |
| `ComputeActivationJacobian(Vector<>)` | Computes the Jacobian matrix of the activation function for a given input vector. |
| `ConvertToOnnx(OnnxGraphBuilder,OnnxLayerInputs)` | Emits this layer as one or more ONNX nodes into `builder`. |
| `CopyTrainableParametersFrom(IReadOnlyList<Tensor<>>)` | Copies the values of `sources` INTO this layer's existing trainable tensors, element for element, without rebinding or allocating — the allocation-free, aliasing-free counterpart to `Tensor{`. |
| `DerivativeTensor(IActivationFunction<>,Tensor<>)` | Calculates the derivative of a scalar activation function for each element of a tensor. |
| `Deserialize(BinaryReader)` | Deserializes the layer's parameters from a binary reader. |
| `Dispose` | Releases all resources used by this layer, including any GPU resources. |
| `Dispose(Boolean)` | Releases resources used by this layer. |
| `DownloadWeightsFromGpu` | Downloads the layer's weights and biases from GPU memory back to CPU. |
| `EnsureInitialized` | Ensures that the layer is initialized. |
| `EnsureInitializedFromInput(Tensor<>)` | Convenience helper for deferred-shape layers: invokes `Tensor{` (if shapes are not yet resolved) followed by `EnsureInitialized`. |
| `EnsureParametersMaterialized` | Forces lazy parameter allocation now (the hook `MaterializeParameters` drives). |
| `EstimateActivationMemory` | Estimates the activation memory (in bytes) needed during a forward pass. |
| `EstimateFlops` | Estimates the computational cost (FLOPs) for a single forward pass through this layer. |
| `Forward(IReadOnlyDictionary<String,Tensor<>>)` | Multi-input forward pass. |
| `Forward(Tensor<>)` | Performs the forward pass of the layer. |
| `Forward(Tensor<>[])` | Performs the forward pass of the layer with multiple input tensors. |
| `ForwardGpu(IReadOnlyDictionary<String,Tensor<>>)` | GPU multi-input forward pass. |
| `ForwardGpu(Tensor<>[])` | Performs the forward pass of the layer on GPU. |
| `ForwardWithPrecisionCheck(Tensor<>)` | Performs a forward pass with automatic mixed-precision handling. |
| `GetActivationTypeFromFunction(Object)` | Gets the standardized activation function type from an activation function object. |
| `GetActivationTypes` | Gets the types of activation functions used by this layer. |
| `GetBiases` | Gets the bias tensor for layers that have trainable biases. |
| `GetBytesPerElement` | Gets the number of bytes per element for the numeric type `T`. |
| `GetDiagnostics` | Gets diagnostic information about this layer's state and behavior. |
| `GetFusedActivationType` | Gets the fused activation type for IEngine fused operations. |
| `GetInputShape` | Gets the input shape for this layer. |
| `GetInputShapes` | Gets all input shapes for this layer. |
| `GetLayerCategory` | Gets the category classification for this layer, used by automated per-layer tools like quantizers, pruners, pipeline partitioners, and LoRA adapters. |
| `GetMetadata` | Returns layer-specific metadata for serialization purposes. |
| `GetOutputShape` | Gets the output shape for this layer. |
| `GetParameterGradients` | Gets the gradients of all trainable parameters in this layer. |
| `GetParameterNames` | Gets all parameter names in this layer. |
| `GetParameterShape(String)` | Gets the expected shape for a parameter. |
| `GetParameters` | Gets all trainable parameters of the layer as a single vector. |
| `GetRegisteredBuffers` | Gets all registered buffers (non-trainable persistent tensors) for this layer. |
| `GetSubLayers` | Returns all child layers registered via `ILayer{`. |
| `GetTrainableParameters` | Returns all trainable parameter tensors registered via `PersistentTensorRole)`. |
| `GetWeights` | Gets the weight matrix for layers that have trainable weights. |
| `HasGpuActivation` | Checks if the layer's scalar activation function supports GPU training. |
| `InitializeLayerBiases(Tensor<>)` | Initializes biases using this layer's `InitializationStrategy`, falling back to zero initialization if none was set. |
| `InitializeLayerWeights(Tensor<>,Int32,Int32)` | Initializes weights using this layer's `InitializationStrategy`, falling back to Xavier/Glorot normal if none was set. |
| `InvalidateTrainableParameter(Tensor<>)` | Notifies the engine that a registered persistent tensor's data has changed. |
| `IsSparseTensor(Tensor<>)` | Returns true when `tensor` is a sparse-storage tensor (CSR/CSC). |
| `LoadWeights(Dictionary<String,Tensor<>>,Func<String,String>,Boolean)` | Loads weights from a dictionary of tensors using optional name mapping. |
| `MapActivationInstanceToFused(Object)` | Maps an activation function instance to its corresponding `FusedActivationType`. |
| `MapActivationToFused` | Maps the layer's activation function to a `FusedActivationType` for GPU-fused operations. |
| `MaterializeParameters` | Forces lazy weight allocation now (the same materialization the first `Forward` performs), so a caller can then read or write the layer's weights by reference through `GetTrainableParameters` / `Tensor{` instead of through a `GetParameters`… |
| `OnFirstForward(Tensor<>)` | Hook fired once when a deferred-shape layer sees its first input. |
| `RegisterBuffer(Tensor<>,String,PersistentTensorRole)` | Registers a non-trainable persistent tensor (buffer) with this layer. |
| `RegisterManualBackwardNode(Tensor<>,Tensor<>[],Func<Tensor<>,Tensor<>[]>)` | Registers a single custom autograd node on the active gradient tape for a layer whose `Tensor{` is a manual (non-Engine-op) computation but which can supply a hand-written backward. |
| `RegisterStreamingWeightsWithPool` | Registers this layer's just-materialized streaming weight tensors with the process-wide `WeightRegistry` so they become evictable LRU entries the moment they exist — not in a batched post-forward sweep. |
| `RegisterSubLayer(ILayer<>)` | Registers a child layer for automatic discovery by the recursive parameter collection system. |
| `RegisterTrainableParameter(Tensor<>,PersistentTensorRole)` | Registers a trainable parameter tensor with the engine for GPU memory optimization. |
| `ResetState` | Resets the internal state of the layer. |
| `ResolveFromShape(Int32[])` | Eagerly resolves a lazy layer's shape (and allocates its weights) from a known input shape, without running an actual forward pass. |
| `ResolveShapes(Int32[],Int32[])` | Resolves a deferred-shape layer's input and output shapes. |
| `ResolveShapesOnly(Int32[])` | Resolves a lazy layer's `InputShape` and `OutputShape` from a concrete input shape WITHOUT allocating or initializing weights. |
| `ReturnPooledParameters` | Hook called by `Boolean)` to return rented parameter tensors to the `TensorAllocator` pool. |
| `RunShapeInference(Action)` | Runs `resolve` with shape-inference mode active, guaranteeing the flag is cleared afterward even on exception. |
| `Serialize(BinaryWriter)` | Serializes the layer's parameters to a binary writer. |
| `SetBiases(Tensor<>)` | Sets the bias tensor for this layer. |
| `SetCaptureMode(Boolean)` | Sets the graph capture mode for this layer. |
| `SetParameter(String,Tensor<>)` | Sets a parameter tensor by name. |
| `SetParameters(Vector<>)` | Sets the trainable parameters of the layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces this layer's trainable parameter tensors with the provided tensors. |
| `SetTrainingMode(Boolean)` | Sets whether the layer is in training mode or inference mode. |
| `SetWeights(Tensor<>)` | Sets the weight tensor for this layer. |
| `ShapeInferenceOutput(Tensor<>)` | Resolves this layer's shape from `input` (without allocating weights) and returns a zero-filled tensor of the resulting forward output shape (`[batch, ...OutputShape]`). |
| `TryDeclareShape` | Proactively declares this layer's parameter shapes WITHOUT requiring a forward pass. |
| `TryGetParameter(String,Tensor<>)` | Tries to get a parameter tensor by name. |
| `UnregisterSubLayer(ILayer<>)` | Removes a previously registered sub-layer. |
| `UnregisterTrainableParameter(Tensor<>)` | Removes a previously-registered trainable parameter tensor from this layer's registration and from the engine's persistent-tensor list. |
| `UpcastResidentWeight(Tensor<>,Tensor<Half>)` | Returns the full-precision (T) form of an fp16-resident weight for use in one matmul, without keeping a full-precision copy resident. |
| `UpdateParameters()` | Updates the parameters of the layer using the calculated gradients. |
| `UpdateParameters(Vector<>)` | Updates the parameters of the layer with the given vector of parameter values. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates the layer's parameters on GPU using the specified optimizer configuration. |
| `UploadWeightsToGpu` | Uploads the layer's weights and biases to GPU memory for GPU-resident training. |
| `ValidateWeights(IEnumerable<String>,Func<String,String>)` | Validates that a set of weight names can be loaded into this layer. |
| `ZeroGrad` | Clears all accumulated gradients. |
| `ZeroGradientsGpu` | Resets the GPU gradient accumulators to zero. |

## Fields

| Field | Summary |
|:-----|:--------|
| `BiasParameterName` | Standard parameter name for bias tensors. |
| `CapturedGraphOutput` | The captured computation graph output node from the last capture-mode Forward() call. |
| `DefaultStrategy` | Initializes a weight tensor using the given strategy, or Xavier uniform by default. |
| `InitializationLock` | Object used for thread-safe lazy initialization. |
| `IsCapturing` | When true, Forward() records operations to a computation graph instead of executing them. |
| `IsTrainingMode` | Gets or sets a value indicating whether the layer is in training mode. |
| `ParameterGradients` | The gradients of the trainable parameters. |
| `Parameters` | The trainable parameters of this layer. |
| `WeightParameterName` | Standard parameter name for weight tensors. |
| `_cachedInputPorts` | Declares the named input ports this layer accepts. |
| `_cachedOutputPorts` | Declares the named output ports this layer produces. |
| `_cachedParameterCount` | Gets the total number of parameters in this layer. |
| `_disposed` | Tracks whether Dispose has been called. |
| `_instanceCounter` | Counter for generating unique instance IDs across all layer instances. |
| `_instanceId` | The unique instance ID for this layer, used to distinguish multiple instances of the same layer type. |
| `_keepActivationCacheUnderTape` | #1624 escape hatch / test hook. |
| `_preActivationCache` | Cached pre-activation input from the most recent ApplyActivation call. |
| `_registeredBuffers` | Non-trainable persistent state tensors registered via `PersistentTensorRole)`. |
| `_registeredSubLayers` | Child layers registered via `ILayer{`. |
| `_registeredTensors` | Collection of tensors that have been registered as persistent with the engine. |
| `s_residentUpcastScratch` | Thread-static, shape-keyed reuse pool for the transient full-precision upcast of an fp16-resident weight (see `Half}@)`). |

