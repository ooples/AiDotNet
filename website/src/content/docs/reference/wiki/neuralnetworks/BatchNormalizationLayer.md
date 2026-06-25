---
title: "BatchNormalizationLayer<T>"
description: "Implements batch normalization for neural networks, which normalizes the inputs across a mini-batch."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Implements batch normalization for neural networks, which normalizes the inputs across a mini-batch.

## For Beginners

Batch normalization is like standardizing test scores in a classroom.

Imagine a class where each student (input) has a raw test score. Batch normalization:

1. Calculates the average score and how spread out the scores are
2. Converts each score to show how many standard deviations it is from the average
3. Applies adjustable scaling and shifting to the standardized scores

This helps neural networks learn more efficiently by:

- Keeping input values in a consistent range
- Reducing the "internal covariate shift" problem
- Making the network less sensitive to poor weight initialization
- Allowing higher learning rates without divergence

In practice, this means your network will typically train faster and perform better.

## How It Works

Batch normalization helps stabilize and accelerate training by normalizing layer inputs.
It works by normalizing each feature to have zero mean and unit variance across the batch,
then applying learnable scale (gamma) and shift (beta) parameters.

Benefits include:

- Faster training convergence
- Reduced sensitivity to weight initialization
- Ability to use higher learning rates
- Acts as a form of regularization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BatchNormalizationLayer(Int32,Double,Double)` | AiDotNet#1370 eager-init constructor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets all trainable parameters of the batch normalization layer. |
| `SupportsGpuExecution` | Gets whether this layer has a GPU implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyInferenceAnyRank(Tensor<>,Tensor<>,Tensor<>)` | Applies batch normalization inference for tensors of any rank. |
| `ConvertToOnnx(OnnxGraphBuilder,OnnxLayerInputs)` | Emits an ONNX `BatchNormalization` op using the layer's running statistics (inference mode). |
| `Forward(Tensor<>)` | Performs the forward pass of batch normalization. |
| `ForwardGpu(Tensor<>[])` | Performs GPU-resident batch normalization forward pass. |
| `GetBeta` | Gets the beta (shift) parameters of the batch normalization layer. |
| `GetEpsilon` | Gets the epsilon value used for numerical stability. |
| `GetGamma` | Gets a value indicating whether this layer supports training mode. |
| `GetMomentum` | Gets the momentum value for running statistics. |
| `GetParameterGradients` | Resets the internal state of the batch normalization layer. |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetRunningMean` | Gets the running mean of the batch normalization layer. |
| `GetRunningVariance` | Gets the running variance of the batch normalization layer. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitializeNormalizationParameters` | Allocates _gamma / _beta / _runningMean / _runningVariance to match the resolved InputShape and registers gamma + beta as trainable. |
| `OnFirstForward(Tensor<>)` | Resolves `numFeatures` on the first forward call by switching on the input rank, allocates gamma/beta + running mean/variance tensors, and registers gamma/beta as trainable parameters. |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SetParameters(Vector<>)` | Sets all trainable parameters of the batch normalization layer. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `SetTrainingMode(Boolean)` | Switches the layer between training and inference behavior. |
| `UpdateParameters()` | Updates the layer's parameters using the computed gradients. |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_beta` | The shift parameter applied after normalization. |
| `_betaGradient` | The gradient of the loss with respect to beta. |
| `_epsilon` | A small constant added to the variance for numerical stability. |
| `_gamma` | The scale parameter applied after normalization. |
| `_gammaGradient` | The gradient of the loss with respect to gamma. |
| `_inputWas1D` | Tracks whether the last forward pass input was rank-1, so backward can preserve rank. |
| `_lastInput` | The input from the last forward pass. |
| `_lastMean` | The batch mean from the last forward pass. |
| `_lastVariance` | The batch variance from the last forward pass. |
| `_momentum` | The momentum for updating running statistics. |
| `_originalInputShape` | Stores the original input shape from forward pass so backward can restore it. |
| `_runningMean` | The running mean used during inference. |
| `_runningVariance` | The running variance used during inference. |
| `_zeroInitGammaPending` | Initializes gamma (scale) parameters to zero. |

