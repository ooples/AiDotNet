---
title: "VAEModelBase<T>"
description: "Base class for Variational Autoencoder (VAE) models used in latent diffusion."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion.VAE`

Base class for Variational Autoencoder (VAE) models used in latent diffusion.

## For Beginners

This is the foundation for all VAE models in the library.
VAEs compress images to a small latent representation and decompress them back.
They are essential for efficient latent diffusion models like Stable Diffusion.

## How It Works

This abstract base class provides common functionality for all VAE implementations,
including encoding, decoding, sampling, and latent scaling operations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VAEModelBase(ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the VAEModelBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` |  |
| `DownsampleFactor` |  |
| `Engine` | Provides access to the hardware-accelerated tensor engine. |
| `InputChannels` |  |
| `LatentChannels` |  |
| `LatentScaleFactor` |  |
| `ParameterCount` |  |
| `SupportsExactGradients` | Whether this VAE supports exact (layer-level) gradients via `Tensor{`. |
| `SupportsParameterInitialization` |  |
| `SupportsSlicing` |  |
| `SupportsTiling` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#ICloneable{AiDotNet#Interfaces#IFullModel{T,AiDotNet#Tensors#LinearAlgebra#Tensor{T},AiDotNet#Tensors#LinearAlgebra#Tensor{T}}}#Clone` |  |
| `ApplyGradients(Vector<>,)` |  |
| `BackpropagateLossGradient(Tensor<>)` | Pushes a loss gradient tensor (shape matching the decoder's output) back through the decoder and encoder layer chain so each layer's parameter gradient cache is populated. |
| `Clone` | Creates a deep copy of the VAE model. |
| `ComputeGradients(Tensor<>,Tensor<>,ILossFunction<>)` |  |
| `ComputeGradientsWithTape(Tensor<>,Tensor<>,Tensor<>[])` | Computes gradients using the Tensors GradientTape for automatic differentiation. |
| `ComputeKLDivergence(Tensor<>,Tensor<>)` | Computes the KL divergence loss for VAE training. |
| `Decode(Tensor<>)` |  |
| `DecodeCompiled(Tensor<>,Func<Tensor<>>)` | Routes `eagerDecode` through the decoder compile host. |
| `DecodeCompiledAsync(Tensor<>,Func<Tensor<>>,CancellationToken)` | Async overload of `Tensor{`. |
| `DeepCopy` |  |
| `Deserialize(Byte[])` |  |
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources held by this VAE. |
| `Encode(Tensor<>,Boolean)` |  |
| `EncodeCompiled(Tensor<>,Func<Tensor<>>)` | Routes `eagerEncode` through the encoder compile host. |
| `EncodeCompiledAsync(Tensor<>,Func<Tensor<>>,CancellationToken)` | Async overload of `Tensor{`. |
| `EncodeWithDistribution(Tensor<>)` |  |
| `EnsureActiveFeatureIndicesInitialized` | Ensures active feature indices are initialized with default values if empty. |
| `ForwardForTraining(Tensor<>)` | Runs the VAE forward pass (encode + decode) without suppressing tape recording. |
| `GetActiveFeatureIndices` |  |
| `GetDynamicShapeInfo` |  |
| `GetFeatureImportance` |  |
| `GetInputShape` |  |
| `GetModelMetadata` |  |
| `GetOutputShape` |  |
| `GetParameterChunks` | Streams the VAE's trainable weight tensors per-tensor without materialising a flat aggregate, mirroring PyTorch's `nn.Module.parameters()` generator pattern. |
| `GetParameterGradients` | Extracts accumulated parameter gradients from all encoder/decoder/norm layers after `Tensor{` has populated them. |
| `GetParameters` |  |
| `InvalidateVAECompiledPlans` | Bump when the layer graph changes (tiling/slicing toggle, weight reassignment) so the compile host drops stale plans on the next EncodeCompiled / DecodeCompiled call. |
| `IsFeatureUsed(Int32)` |  |
| `LoadModel(String)` |  |
| `LoadState(Stream)` |  |
| `Predict(Tensor<>)` |  |
| `Sample(Tensor<>,Tensor<>,Nullable<Int32>)` |  |
| `SampleNoise(Int32[],Random)` | Samples random noise from a standard normal distribution. |
| `SanitizeParameters(Vector<>)` |  |
| `SaveModel(String)` |  |
| `SaveState(Stream)` |  |
| `ScaleLatent(Tensor<>)` |  |
| `Serialize` |  |
| `SetActiveFeatureIndices(IEnumerable<Int32>)` |  |
| `SetParameters(Vector<>)` |  |
| `SetSlicingEnabled(Boolean)` |  |
| `SetTilingEnabled(Boolean)` |  |
| `ThrowIfDisposed` | Throws `ObjectDisposedException` when this VAE has already been disposed. |
| `Train(Tensor<>,Tensor<>)` |  |
| `TryShareParametersFrom(VAEModelBase<>)` | COW clone lever (#1624): shares each trainable weight tensor's STORAGE with `source` via the global `CopyOnWriteCloneHelper` (O(1)-until-write), instead of the flat `GetParameters()` → `SetParameters()` round-trip that materializes the enti… |
| `UnscaleLatent(Tensor<>)` |  |
| `WithParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `LossFunction` | The loss function used for training. |
| `NumOps` | Provides numeric operations for the specific type T. |
| `RandomGenerator` | Random number generator for sampling operations. |
| `SlicingEnabled` | Whether slicing mode is enabled for sequential processing. |
| `TilingEnabled` | Whether tiling mode is enabled for memory-efficient processing. |
| `_activeFeatureIndices` | Active feature indices used by the model. |
| `_encoderCompileHost` | Compile host shared by Encode and Decode forward paths so VAE inference traces once and replays the compiled plan on subsequent calls. |

