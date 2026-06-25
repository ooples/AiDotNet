---
title: "DiffusionConvLayer<T>"
description: "Implements diffusion convolution for mesh surface processing using the heat diffusion equation."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements diffusion convolution for mesh surface processing using the heat diffusion equation.

## For Beginners

Think of heat spreading on a surface:

- Place a heat source at each vertex
- Let the heat diffuse across the mesh surface
- After some time, nearby vertices (in geodesic distance) will share heat
- Use this diffusion pattern to aggregate features from neighbors

Key advantages:

- Respects the true surface geometry (not just mesh connectivity)
- Adaptive receptive field based on diffusion time
- Robust to mesh discretization and vertex density

Applications:

- Shape classification and segmentation
- Surface analysis (curvature, normals, features)
- Correspondence between shapes
- Texture synthesis on surfaces

## How It Works

DiffusionConvLayer applies learned diffusion kernels on mesh surfaces to aggregate
information across geodesic neighborhoods. Instead of using fixed spatial neighborhoods,
it leverages the heat diffusion equation to create adaptive receptive fields that
respect the underlying geometry.

Reference: "DiffusionNet: Discretization Agnostic Learning on Surfaces" by Sharp et al.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiffusionConvLayer(Int32,Int32,Int32,IActivationFunction<>,Nullable<Boolean>)` | Initializes a new instance of the `DiffusionConvLayer` class. |
| `DiffusionConvLayer(Int32,Int32,Int32,IVectorActivationFunction<>,Nullable<Boolean>)` | Initializes a new instance of the `DiffusionConvLayer` class with vector activation. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DiffusionTimes` | Gets the learned diffusion time values. |
| `InputChannels` | Gets the number of input feature channels per vertex. |
| `NumTimeScales` | Gets the number of diffusion time scales to use. |
| `OutputChannels` | Gets the number of output feature channels per vertex. |
| `ParameterCount` | Gets the total number of trainable parameters. |
| `SupportsGpuExecution` | Gets a value indicating whether this layer supports GPU-accelerated forward pass. |
| `SupportsGpuTraining` | Gets a value indicating whether this layer supports GPU training. |
| `SupportsTraining` | Gets a value indicating whether this layer supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddBiases(Tensor<>,Int32)` | Adds biases to each vertex output. |
| `BackpropagateDiffusionDirectVectorized(Tensor<>,Int32)` | Backpropagates through direct diffusion using vectorized operations. |
| `BackpropagateDiffusionSpectralVectorized(Tensor<>,Int32)` | Backpropagates through spectral diffusion using vectorized operations. |
| `BackpropagateThroughDiffusion(Tensor<>,Int32)` | Backpropagates gradients through the diffusion operation using vectorized operations. |
| `Clone` | Creates a deep copy of this layer. |
| `ComputeDiffusedFeatures(Tensor<>,Int32)` | Computes diffused features at multiple time scales. |
| `ComputeDiffusionDirect(Tensor<>,[],Int32)` | Computes diffusion using direct matrix method with vectorized operations. |
| `ComputeDiffusionDirectSingleTime(Tensor<>,[],Int32,Int32)` | Computes direct diffusion for a single time scale. |
| `ComputeDiffusionSpectral(Tensor<>,[],Int32)` | Computes diffusion using spectral method (eigenbasis) with vectorized operations. |
| `ComputeDiffusionTimeGradients(Tensor<>,Tensor<>,Int32)` | Computes gradients for diffusion time parameters using the chain rule. |
| `ComputeTimeGradientsDirect(Tensor<>,Tensor<>,Int32)` | Computes time gradients using direct (iterative) method. |
| `ComputeTimeGradientsSpectral(Tensor<>,Tensor<>,Int32)` | Computes time gradients using spectral method. |
| `Deserialize(BinaryReader)` | Deserializes the layer from a binary stream. |
| `ExtractBatchSlice(Tensor<>,Int32,Int32)` | Extracts a single sample from a batched tensor. |
| `Forward(Tensor<>)` | Performs the forward pass of diffusion convolution. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-accelerated forward pass using spectral heat diffusion. |
| `GetBiases` | Gets the bias tensor. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` | Gets all trainable parameters as a single vector. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GetWeights` | Gets the weight tensor. |
| `InitializeDiffusionTimes(Int32)` | Initializes diffusion times as log-spaced values. |
| `InitializeWeights` | Initializes weights using He initialization. |
| `OnFirstForward(Tensor<>)` | Resolves input channels and vertex count from input.Shape on first forward (rank-2 [V, C] or rank-3 [B, V, C]) and allocates weights+biases. |
| `ProcessBatched(Tensor<>,Int32,Int32)` | Processes a batched input tensor. |
| `ProcessSingle(Tensor<>,Int32)` | Processes a single (non-batched) input tensor. |
| `ResetState` | Resets cached state from forward/backward passes. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `Serialize(BinaryWriter)` | Serializes the layer to a binary stream. |
| `SetEigenbasis([],Tensor<>,Tensor<>)` | Sets the Laplacian eigenbasis for the current mesh. |
| `SetLaplacian(Tensor<>,Tensor<>)` | Sets the Laplacian matrix directly for the current mesh. |
| `SetParameters(Vector<>)` | Sets all trainable parameters from a vector. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` | Updates layer parameters using computed gradients. |
| `UpdateParametersGpu(IGpuOptimizerConfig)` | Updates parameters on GPU using the configured optimizer. |
| `ValidateParameters(Int32,Int32,Int32,Int32)` | Validates constructor parameters. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_biases` | Learnable bias values [OutputChannels]. |
| `_biasesGradient` | Cached bias gradients from backward pass. |
| `_diffusedFeatures` | Cached diffused features for backward pass [numVertices, InputChannels * NumTimeScales]. |
| `_diffusionTimesGradient` | Cached gradients for diffusion time parameters from backward pass. |
| `_eigenbasisLock` | Synchronizes automatic eigenbasis computation. |
| `_eigenvalues` | Eigenvalues of the Laplacian [numEigenvalues]. |
| `_eigenvectors` | Eigenvectors of the Laplacian [numVertices, numEigenvalues]. |
| `_gpuBiases` | GPU bias tensor. |
| `_gpuBiasesGradient` | GPU bias gradients. |
| `_gpuBiasesVelocity` | GPU optimizer state for biases. |
| `_gpuDiffusedFeatures` | Cached GPU diffused features for backward pass. |
| `_gpuDiffusionTimes` | GPU diffusion time tensor. |
| `_gpuDiffusionTimesGradient` | GPU diffusion time gradients. |
| `_gpuDiffusionTimesVelocity` | GPU optimizer state for diffusion times. |
| `_gpuInput` | Cached GPU input from the last forward pass. |
| `_gpuInputShape` | Cached GPU input shape from the last forward pass. |
| `_gpuOutput` | Cached GPU activated output for backward pass. |
| `_gpuPreActivation` | Cached GPU pre-activation output for backward pass. |
| `_gpuWeights` | GPU weight tensor. |
| `_gpuWeightsGradient` | GPU weight gradients. |
| `_gpuWeightsVelocity` | GPU optimizer state for weights. |
| `_laplacian` | Laplacian matrix for the current mesh [numVertices, numVertices]. |
| `_lastInput` | Cached input from the last forward pass. |
| `_lastOutput` | Cached output from the last forward pass. |
| `_lastPreActivation` | Cached pre-activation output from the last forward pass. |
| `_massMatrix` | Mass matrix (vertex areas) for the current mesh [numVertices]. |
| `_numEigenvectors` | Number of eigenvectors to use for spectral acceleration. |
| `_preferSpectralDiffusion` | Controls automatic eigenbasis computation for CPU execution. |
| `_weights` | Learnable weights [OutputChannels, InputChannels * NumTimeScales]. |
| `_weightsGradient` | Cached weight gradients from backward pass. |

