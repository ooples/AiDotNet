---
title: "GenerativeAdversarialNetwork<T>"
description: "Represents a Generative Adversarial Network (GAN), a deep learning architecture that consists of two neural networks (a generator and a discriminator) competing against each other in a zero-sum game."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks`

Represents a Generative Adversarial Network (GAN), a deep learning architecture that consists of two neural networks
(a generator and a discriminator) competing against each other in a zero-sum game.

## For Beginners

A GAN is like an art forger and an art detective competing against each other.

Think of it this way:

- The generator is like an art forger trying to create fake paintings that look real
- The discriminator is like an art detective trying to tell which paintings are real and which are fake
- As the forger gets better, the detective has to improve too
- As the detective gets better, the forger is forced to create more convincing fakes
- Eventually, the forger becomes so good that their fake paintings are nearly indistinguishable from real ones

This continuous competition drives both networks to improve, resulting in a generator that
can create remarkably realistic synthetic data like images, music, or text.

## How It Works

A Generative Adversarial Network (GAN) is a powerful machine learning architecture that uses two neural networks - 
a generator and a discriminator - that are trained simultaneously through adversarial training. The generator 
network learns to create realistic synthetic data samples (like images), while the discriminator network learns 
to distinguish between real data and the generator's synthetic outputs. As training progresses, the generator 
becomes better at creating realistic data, and the discriminator becomes better at distinguishing real from fake, 
pushing each other to improve in a competitive process.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GenerativeAdversarialNetwork(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,GenerativeAdversarialNetworkOptions)` | Initializes a new instance of the `GenerativeAdversarialNetwork` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuxiliaryLossWeight` | Gets or sets the weight for auxiliary losses (gradient penalty, feature matching). |
| `Discriminator` | Gets the discriminator network that distinguishes between real and synthetic data. |
| `DiscriminatorOptimizer` | Gets the optimizer used for updating discriminator parameters. |
| `FeatureMatchingLayers` | Gets or sets the indices of discriminator layers to use for feature matching. |
| `FeatureMatchingWeight` | Gets or sets the weight applied to the feature matching loss. |
| `Generator` | Gets the generator network that creates synthetic data. |
| `GeneratorOptimizer` | Gets the optimizer used for updating generator parameters. |
| `ParameterCount` | Updates the parameters of both the Generator and Discriminator networks. |
| `UseAuxiliaryLoss` | Gets or sets whether to use auxiliary losses (gradient penalty, feature matching) during training. |
| `UseFeatureMatching` | Gets or sets whether feature matching is enabled for generator training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildTapeTrackedGradientPenalty(Tensor<>,Tensor<>,Double)` | Computes the gradient penalty for WGAN-GP (Wasserstein GAN with Gradient Penalty). |
| `CalculateBatchGradients(Tensor<>,Tensor<>)` | Calculates gradients for backpropagation from predictions and targets. |
| `CalculateBatchLoss(Tensor<>,Tensor<>)` | Calculates the loss for a batch of predictions and target values. |
| `ComputeAuxiliaryLoss` | Computes the auxiliary loss for the GAN, which includes gradient penalty and feature matching losses. |
| `ComputeBatchMean(Tensor<>)` | Computes the mean of a tensor across the batch dimension. |
| `ComputeFeatureMatchingLoss` | Computes the feature matching loss between real and generated data. |
| `ComputeGradientPenalty` | Computes the gradient penalty for WGAN-GP using the most recent real / fake batches captured during training. |
| `ComputeNumericalGradient(Tensor<>)` | Computes numerical gradients of discriminator output with respect to input using finite differences. |
| `ComputeSymbolicGradient(Tensor<>)` | Computes gradients of discriminator output with respect to input using symbolic differentiation. |
| `CreateGANArchitecture(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType)` | Creates the combined GAN architecture from generator and discriminator architectures. |
| `CreateLabelTensor(Int32,)` | Creates a tensor filled with a specified value, typically used for labels. |
| `CreateNetworkForInputType(NeuralNetworkArchitecture<>,InputType,ILossFunction<>)` | Creates the appropriate neural network type based on the input type. |
| `CreateNewInstance` | Creates a new instance of the GenerativeAdversarialNetwork with the same configuration as the current instance. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes GAN-specific data from a binary reader. |
| `DiscriminateImages(Tensor<>)` | Evaluates how real a batch of images appears to the discriminator. |
| `EnableFeatureMatching(Boolean)` | Enables feature matching loss to encourage the generator to match statistics of real data. |
| `EnableGradientPenalty(Boolean)` | Enables gradient penalty (WGAN-GP) for improved training stability. |
| `EnableGradientPenalty(Boolean,Double)` | Enables WGAN-GP gradient penalty and overrides the penalty coefficient. |
| `EvaluateModel(Int32)` | Evaluates the GAN by generating a batch of images and calculating metrics for their quality. |
| `EvaluateModelWithTensors(Int32)` | Evaluates the GAN using tensor operations. |
| `GenerateImages(Tensor<>)` | Generates synthetic images using tensor operations. |
| `GenerateQualityImages(Int32,Double)` | Generates high-quality images by filtering based on discriminator scores. |
| `GenerateRandomNoiseTensor(Int32,Int32)` | Generates a tensor of random noise for the generator. |
| `GetAuxiliaryLossDiagnostics` | Gets diagnostic information about the auxiliary losses. |
| `GetDiagnostics` | Gets diagnostic information about this component's state and behavior. |
| `GetModelMetadata` | Gets metadata about the GAN model, including information about both generator and discriminator components. |
| `GetNamedLayerActivations(Tensor<>)` | Gets named layer activations from the generator network. |
| `GetOptions` |  |
| `GetParameterChunks` |  |
| `GetParameters` | Gets the combined parameters from both the generator and discriminator networks. |
| `InitializeLayers` | Initializes the layers of the Generative Adversarial Network. |
| `PredictCore(Tensor<>)` | Performs a forward pass through the generator network using a tensor input. |
| `ResetOptimizerState` | Resets the optimizer state to its initial values. |
| `SetTrainingMode(Boolean)` | Propagates training/eval mode to the Generator and Discriminator sub-networks. |
| `Train(Tensor<>,Tensor<>)` | Trains both the generator and discriminator using tensor-based operations throughout. |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one step of training for both the generator and discriminator using tensor batches. |
| `UpdateNetworkParameters(NeuralNetworkBase<>)` | Updates the parameters of a network using the configured optimizer with the calculated gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `GanSerializationVersion` | Serializes GAN-specific data to a binary writer. |
| `_discriminatorOptimizer` | The optimizer used for updating discriminator parameters during training. |
| `_generatorLosses` | Gets or sets the list of recent generator loss values for monitoring training progress. |
| `_generatorOptimizer` | The optimizer used for updating generator parameters during training. |
| `_gradientPenaltyLambda` | WGAN-GP penalty coefficient λ. |
| `_lastDiscriminatorLoss` | Stores the last discriminator loss for diagnostics. |
| `_lastFakeBatch` | Stores the last fake batch for feature matching computation. |
| `_lastFeatureMatchingLoss` | Stores the last computed feature matching loss for diagnostics. |
| `_lastGeneratorLoss` | Stores the last generator loss for diagnostics. |
| `_lastGradientPenalty` | Stores the last computed gradient penalty value for diagnostics. |
| `_lastRealBatch` | Stores the last real batch for feature matching computation. |
| `_useGradientPenalty` | Gets or sets whether gradient penalty (WGAN-GP) is enabled for training stability. |

