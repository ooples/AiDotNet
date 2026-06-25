---
title: "DiffusionModelBase<T>"
description: "Base class for diffusion-based generative models providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion`

Base class for diffusion-based generative models providing common functionality.

## For Beginners

This is the foundation that all diffusion models build upon.
It handles the common tasks that every diffusion model needs:

- The generation loop (iteratively denoising from noise)
- Adding noise during training
- Computing the training loss
- Saving and loading the model

Specific diffusion models (like DDPM, Latent Diffusion) extend this base to implement
their unique noise prediction architectures.

## How It Works

This abstract base class implements the common behavior for all diffusion models,
including the generation loop, noise addition, loss computation, and state management.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionModelBase(DiffusionModelOptions<>,INoiseScheduler<>,NeuralNetworkArchitecture<>)` | Initializes a new instance of the DiffusionModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Architecture` | Gets the optional neural network architecture used for custom layer configuration. |
| `DefaultLossFunction` |  |
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `IsQuantizationAwareTrainingEnabled` | Whether quantization-aware training is engaged for `Tensor{` (G5, #1624). |
| `Options` | Gets the configuration options for this model. |
| `ParameterCount` |  |
| `PredictorStreamCapture` | Runs one denoising-step noise prediction, optionally inside a GPU deferred execution graph (AiDotNet.Tensors #642) when `UseGpuExecutionGraph` is enabled and the active engine is a CUDA `DirectGpuTensorEngine`. |
| `QatThresholdOverride` | Test/diagnostic override for `DefaultQatThresholdParams`. |
| `Scheduler` |  |
| `SupportsParameterInitialization` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#ICloneable{AiDotNet#Interfaces#IFullModel{T,AiDotNet#Tensors#LinearAlgebra#Tensor{T},AiDotNet#Tensors#LinearAlgebra#Tensor{T}}}#Clone` |  |
| `ApplyGradients(Vector<>,)` |  |
| `Clone` | Creates a deep copy of the model. |
| `CollectTrainableLayers` | Reflection-walks the model graph (same traversal as `CollectTrainableParameters`) and returns every `ITrainableLayer` in a stable order, so a clone's layers can be paired positionally with its parent's. |
| `CollectTrainableParameters` | Collects all trainable parameter tensors from the noise predictor's layers. |
| `ComputeGradients(Tensor<>,Tensor<>,ILossFunction<>)` |  |
| `ComputeLoss(Tensor<>,Tensor<>,Int32[])` |  |
| `CreateInferenceRng(Nullable<Int32>)` | Creates a reproducible RNG for the INFERENCE / generation path. |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `DisableQuantizationAwareTraining` | Forces quantization-aware training OFF, overriding the parameter-count default. |
| `Dispose` |  |
| `Dispose(Boolean)` | Cascades Dispose to every disposable component the concrete model exposes via `EnumerateDisposableComponents` (default: reflection walk over instance fields), plus the owned `_scheduler`. |
| `EnableQuantizationAwareTraining(QuantizationConfiguration)` | Forces quantization-aware training ON (G5, #1624): from the next `Tensor{` step the forward pass uses fake-quantized weights (quantize→dequantize, simulating int8 inference precision) while keeping full-precision shadow weights that the opt… |
| `EnsureActiveFeatureIndicesInitialized` | Ensures active feature indices are initialized with default values if empty. |
| `EnsureOwnWeights` | Copy-on-write guard: if this model is sharing its weight tensors with another model, give it a private deep copy of every weight tensor BEFORE the caller mutates any of them. |
| `EnumerateDisposableComponents` | Concrete diffusion models can override this method to yield the components they own that hold disposable resources — typically the noise predictor (DiT, UNet, MMDiT) plus, for latent diffusion, the VAE and conditioner. |
| `FlattenGradients(Tensor<>[],Dictionary<Tensor<>,Tensor<>>)` | Flattens gradient tensors into a single vector matching GetParameters() layout. |
| `Generate(Int32[],Int32,Nullable<Int32>)` |  |
| `GenerateAsync(Int32[],Int32,Nullable<Int32>,CancellationToken)` | Async overload of `Int32})`. |
| `GenerateAsyncCore(Int32[],Int32,Nullable<Int32>,Vector<>,CancellationToken)` | True-async denoising loop. |
| `GetActiveFeatureIndices` |  |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` |  |
| `GetInputShape` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `GetOutputShape` |  |
| `GetParameterChunks` | Streams the diffusion stack's trainable weight tensors per-tensor. |
| `GetParameters` |  |
| `InvalidateTrainableParametersCache` | Invalidates the cached trainable-parameter walk. |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `Predict(Tensor<>)` |  |
| `PredictNoise(Tensor<>,Int32)` |  |
| `PredictNoiseAsync(Tensor<>,Int32,Tensor<>,CancellationToken)` | Concrete async noise prediction overload for the base path. |
| `ResolveInitialSample(Int32[],Int32,Nullable<Int32>,Vector<>,Int64)` | Resolves the initial denoising sample: if the caller passed an explicit `initialSample`, validates its length and returns it directly; otherwise draws fresh Gaussian noise via the seeded RNG. |
| `SampleNoise(Int32,Random)` | Samples random noise from a standard normal distribution. |
| `SanitizeNonFiniteElements(Vector<>)` | Replaces every NaN / Infinity element of `sample` with zero (Ho et al. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameterChunks(IEnumerable<Tensor<>>)` | Streaming counterpart to `Vector{`: assigns weights from per-tensor chunks in `GetParameterChunks` order without materializing a flat aggregate. |
| `SetParameters(Vector<>)` |  |
| `ShareWeightsFrom(DiffusionModelBase<>)` | Makes this (freshly-constructed) model SHARE `parent`'s weight tensors by reference — the copy-on-write fast path for `Clone`. |
| `TryShareParametersFrom(DiffusionModelBase<>)` | COW clone lever (#1624): shares each trainable weight tensor's STORAGE with `source` via the Tensors copy-on-write `Tensor<T>.CloneShared()` (O(1)-until-write), instead of the flat `GetParameters()` → `SetParameters()` round-trip that mater… |
| `ValidateGenerateInputs(Int32[],Int32,Int64)` | Internal Generate that optionally accepts an explicit starting sample. |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultQatThresholdParams` | Parameter-count threshold at/above which QAT engages by DEFAULT (G5, #1624). |
| `InferenceDefaultSeed` | Fixed fallback seed for `Int32})` when the model was constructed without a seed — any constant works; it only needs to be stable across calls so unseeded generation is reproducible. |
| `LearningRate` | The learning rate converted to type T for training computations. |
| `LossFunction` | The loss function used for training (typically MSE for noise prediction). |
| `NumOps` | Provides numeric operations for the specific type T. |
| `RandomGenerator` | Random number generator for noise sampling. |
| `_activeFeatureIndices` | Active feature indices used by the model. |
| `_architecture` | The optional neural network architecture blueprint for custom layer configuration. |
| `_cachedTrainableParameters` | Cached result of the reflection walk that discovers trainable parameter tensors. |
| `_options` | The configuration options for this diffusion model. |
| `_qatConfig` | QAT config (explicit or default). |
| `_qatExplicit` | Explicit QAT override: `null` = auto (engage by the parameter-count threshold), `true`/`false` = forced on/off. |
| `_qatHook` | G5 (#1624) quantization-aware-training hook, created lazily on the first `Tensor{` step once `IsQuantizationAwareTrainingEnabled` is true. |
| `_scheduler` | The step scheduler controlling the diffusion process. |

