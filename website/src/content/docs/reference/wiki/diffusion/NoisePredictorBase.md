---
title: "NoisePredictorBase<T>"
description: "Base class for noise prediction networks used in diffusion models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion.NoisePredictors`

Base class for noise prediction networks used in diffusion models.

## For Beginners

This is the foundation that all noise prediction networks build upon.
Noise predictors are the neural networks at the heart of diffusion models that learn to
predict what noise was added to a sample. Different architectures (U-Net, DiT, etc.)
extend this base class.

## How It Works

This abstract base class provides common functionality for all noise predictors,
including timestep embedding, parameter management, serialization, and gradient computation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NoisePredictorBase(ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the NoisePredictorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ActivationCheckpointingEnabled` | Whether this predictor recomputes block activations during backward instead of storing them (activation checkpointing — the standard transformer memory/compute trade). |
| `BaseChannels` |  |
| `CompiledMultiInputReplays` | Count of successful multi-input compiled-plan executions on this predictor's compile host (the per-step denoising path). |
| `ContextDimension` |  |
| `DefaultLossFunction` |  |
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `InputChannels` |  |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `StreamingResidentCapOverride` | Test/diagnostic override for the streaming pool's resident-byte cap so a Test/diagnostic override for the streaming pool's resident-byte cap so a small model can be forced to page (the auto-cap is sized for foundation models). |
| `StreamingThresholdOverride` | Test/diagnostic override for `DefaultStreamingThresholdParams` so controlled-scale tests can exercise the streaming path without a foundation-scale model. |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `SupportsParameterInitialization` |  |
| `TimeEmbeddingDim` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#ICloneable{AiDotNet#Interfaces#IFullModel{T,AiDotNet#Tensors#LinearAlgebra#Tensor{T},AiDotNet#Tensors#LinearAlgebra#Tensor{T}}}#Clone` |  |
| `ApplyGradients(Vector<>,)` |  |
| `BeginWeightStreamingForward` | Starts a predictor forward that may need transparent weight streaming. |
| `CheckpointBlocks(Func<Tensor<>,Tensor<>>[],Tensor<>)` | Runs a sequence of residual-stream blocks under activation checkpointing when `ActivationCheckpointingEnabled` is set, otherwise eagerly. |
| `ChunkOf(DenseLayer<>)` | Wraps a layer's flat parameter vector in a 1-D `Tensor` chunk for the streaming `GetParameterChunks` contract (#1624). |
| `Clone` | Creates a deep copy of the noise predictor. |
| `ComputeGradients(Tensor<>,Tensor<>,ILossFunction<>)` |  |
| `ComputeGradientsWithTape(Tensor<>,Tensor<>,Tensor<>[],Func<Tensor<>,Tensor<>,Tensor<>>,Func<Tensor<>,Tensor<>>)` | Computes gradients using the engine's `GradientTape` for automatic differentiation. |
| `ComputeResidentCapBytes` | Resident-byte cap for the streaming pool: half the host's available managed memory, clamped to [512 MiB, 8 GiB]. |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases managed resources — compiled plans first (so pooled tensor buffers the plans captured are freed before layers Dispose and return their weights), then every `ILayer` exposed by `EnumerateLayers` that implements `IDisposable`. |
| `EagerLayerNorm(Int32)` | Creates a `LayerNormalizationLayer` pre-resolved against `featureSize` so its gamma/beta tensors are fully allocated at construction time. |
| `EnsureActiveFeatureIndicesInitialized` | Ensures active feature indices are initialized with default values if empty. |
| `EnumerateLayers` | Concrete predictors can override to expose their `ILayer` instances for (a) Dispose cascade — pool-rented weight tensors return to the allocator, and (b) future compilation features (plan serialization, CUDA Graph capture) that need visibil… |
| `Forward(Tensor<>)` | Forward pass through the noise predictor's layers, used as the differentiable path for tape-based gradient computation in `Tensor{`. |
| `GetActiveFeatureIndices` |  |
| `GetAvailableMemoryBytesOrZero` | Best-effort available host memory in bytes, used by the memory-aware streaming-engagement heuristic (`MaybeEngageWeightStreaming`). |
| `GetDeclaredWalkableFields(Type)` | Reference-type, non-string instance fields declared directly on `declaringType`. |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` |  |
| `GetInputShape` |  |
| `GetModelMetadata` |  |
| `GetOutputShape` |  |
| `GetParameterChunks` | Streams the predictor's trainable weight tensors per-tensor without materialising a flat aggregate, mirroring PyTorch's `nn.Module.parameters()` generator pattern. |
| `GetParameterGradients` | Extracts accumulated parameter gradients from all layers after backpropagation. |
| `GetParameters` |  |
| `GetTimestepEmbedding(Int32)` |  |
| `InvalidateCompiledPlans` | Bump to signal the layer graph has changed — lazy init expanded a tensor, weights were reassigned, a sub-layer was replaced. |
| `IsFeatureUsed(Int32)` |  |
| `IsWalkableWrapper(Type)` | Returns true for reference types that look like AiDotNet-internal wrapper objects worth descending into during layer enumeration. |
| `LazyConv2D(Int32,Int32,Int32,Int32,Int32,Int32,Int32,IActivationFunction<>)` | Creates a `ConvolutionalLayer` with lazy weight allocation. |
| `LazyDense(Int32,Int32,IActivationFunction<>)` | Creates a `DenseLayer` with lazy weight allocation — weight/bias tensors stay zero-sized until the first Forward() call. |
| `LazyDenseVec(Int32,Int32,IVectorActivationFunction<>)` | Creates a `DenseLayer` with a vector activation and lazy weight allocation. |
| `LazyDenseZero(Int32,Int32)` | Creates a lazily-allocated `DenseLayer` whose weights and biases zero-fill on first resolve (the `Zero` strategy is non-lazy, so the deferred `EnsureInitialized` applies it). |
| `LazyMHA(Int32,Int32,Int32,IActivationFunction<>)` | Creates a `MultiHeadAttentionLayer` with lazy Q/K/V/O weight allocation. |
| `LazySelfAttention(Int32,Int32,Int32,IActivationFunction<>)` | Creates a `SelfAttentionLayer` with lazy Q/K/V weight allocation. |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `MaybeEngageWeightStreaming` | Engages transparent weight streaming for this predictor when it is large enough to pressure host RAM. |
| `Predict(Tensor<>)` |  |
| `PredictCompiled(Tensor<>,Func<Tensor<>>)` | Runs `eagerFallback` under the compile host — traces on first call at each input shape, replays the compiled plan on subsequent calls. |
| `PredictCompiledAsync(Tensor<>,Func<Tensor<>>,CancellationToken)` | Async overload of `Tensor{` — routes through `CancellationToken)` so the compiled plan's `ExecuteAsync` path is taken. |
| `PredictCompiledMulti(Tensor<>[],Func<Tensor<>>)` | Multi-input compiled replay for a forward that reads SEVERAL per-call-varying leaves — the diffusion per-step `Tensor{` reads the noisy sample, the per-step timestep embedding, and optional conditioning. |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` |  |
| `PredictNoiseAsync(Tensor<>,Int32,Tensor<>,CancellationToken)` | Async overload of `Tensor{`. |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `ReflectInstanceLayers(Object)` | Walks an object's instance fields and yields anything that implements `ILayer`, recursively descending into owned wrapper objects (e.g. |
| `RegisterResolvedStreamingWeights` | Registers this predictor's now-resolved weights with the streaming pool, dropping their resident in-memory copies to disk-backed storage. |
| `SampleNoise(Int32[],Random)` | Samples random noise from a standard normal distribution. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetChunk(IEnumerator<Tensor<>>,DenseLayer<>)` | Pulls the next chunk from `e` and assigns it to `layer`, used by streaming `SetParameterChunks` (#1624). |
| `SetParameterChunks(IEnumerable<Tensor<>>)` | Streaming counterpart to `Vector{`: assigns the predictor's weights from per-tensor chunks in `GetParameterChunks` order without materializing a flat aggregate. |
| `SetParameters(Vector<>)` |  |
| `ThrowIfDisposed` | Throws `ObjectDisposedException` when the predictor has already been disposed. |
| `Train(Tensor<>,Tensor<>)` |  |
| `TryShareParametersFrom(NoisePredictorBase<>)` | COW clone lever (#1624): shares each trainable weight tensor's STORAGE with `source` via the global `CopyOnWriteCloneHelper` (O(1)-until-write), instead of the flat `GetParameters()` → `SetParameters()` round-trip that materializes the enti… |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultStreamingThresholdParams` | Parameter-count threshold above which `MaybeEngageWeightStreaming` engages disk-backed weight streaming for the denoising forward loop. |
| `LossFunction` | The loss function used for training (typically MSE for noise prediction). |
| `NumOps` | Provides numeric operations for the specific type T. |
| `RandomGenerator` | Random number generator for initialization and stochastic operations. |
| `_activeFeatureIndices` | Active feature indices used by the model. |
| `_compileHost` | Composable inference-compilation helper. |
| `_inferenceGate` | Verify-then-trust gate (#1622 L3b) shared with `NeuralNetworkBase`: the per-step `Tensor{` call a diffusion denoising loop makes runs eager once per shape to confirm the compiled plan matches, then replays the trusted plan for the remaining… |
| `_layerStructureVersion` | Monotonic layer-graph version. |
| `_timestepEmbeddingCache` | Cache for timestep embeddings to avoid recomputing sinusoidal embeddings for the same timestep during the denoising loop. |
| `s_autoCompiledInferenceEnabled` | Auto-compiled inference replay (#1622) is OPT-IN for noise predictors and OFF by default. |

