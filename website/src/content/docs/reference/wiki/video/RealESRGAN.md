---
title: "RealESRGAN<T>"
description: "Real-ESRGAN (Real Enhanced Super-Resolution GAN) for image and video super-resolution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video`

Real-ESRGAN (Real Enhanced Super-Resolution GAN) for image and video super-resolution.

## For Beginners

Real-ESRGAN upscales images and video frames to higher resolution
while adding realistic details. It's one of the most practical super-resolution models.

The network works by:

1. Extracting deep features from low-resolution input using RRDB blocks
2. Upsampling using pixel shuffle (efficient sub-pixel convolution)
3. Training with adversarial loss to add realistic textures
4. Using perceptual loss to ensure visual quality

Example usage:

## How It Works

Real-ESRGAN is a practical super-resolution model that uses:

- RRDB (Residual in Residual Dense Block) generator for deep feature extraction
- U-Net discriminator for adversarial training
- Combined loss: L1 (pixel) + Perceptual (VGG) + GAN (adversarial)
- Second-order degradation model for realistic training data

**Reference:** Wang et al., "Real-ESRGAN: Training Real-World Blind Super-Resolution
with Pure Synthetic Data", ICCV 2021. https://arxiv.org/abs/2107.10833

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RealESRGAN(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,Int32,Int32,Int32,Double,Double,Double,Double,RealESRGANOptions)` | Initializes a new instance of the Real-ESRGAN class. |
| `RealESRGAN(NeuralNetworkArchitecture<>,String,Int32,RealESRGANOptions)` | Creates a Real-ESRGAN model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Discriminator` | Gets the U-Net discriminator network that judges image quality. |
| `DiscriminatorRequired` | Gets the discriminator, throwing if not initialized (native mode only). |
| `Generator` | Gets the RRDB-Net generator network that produces super-resolved images. |
| `GeneratorRequired` | Gets the generator, throwing if not initialized (native mode only). |
| `LastDiscriminatorLoss` | Gets the last discriminator loss value. |
| `LastGeneratorLoss` | Gets the last generator loss value. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UpscaleFactor` | Gets the upscaling factor for this model. |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateBCEGradient(Tensor<>,Tensor<>)` | Calculates the gradient of binary cross-entropy loss. |
| `CalculateReconstructionGradient(Tensor<>,Tensor<>)` | Calculates the reconstruction (L1) loss gradient. |
| `CombineGradients(Tensor<>,Tensor<>)` | Combines multiple gradient tensors. |
| `CreateLabelTensor(Int32,)` | Creates a tensor filled with a specified value for labels. |
| `CreateNewInstance` |  |
| `CreateRealESRGANArchitecture(NeuralNetworkArchitecture<>,InputType)` | Creates the combined Real-ESRGAN architecture from generator and discriminator architectures. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `PostprocessOutput(Tensor<>)` |  |
| `PredictCore(Tensor<>)` |  |
| `PredictOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `PreprocessFrames(Tensor<>)` |  |
| `ProcessThroughDiscriminator(Tensor<>)` | Processes a tensor that may have a batch dimension through the discriminator. |
| `ProcessThroughGenerator(Tensor<>)` | Processes a tensor that may have a batch dimension through the generator. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ShapesMatch(Int32[],Int32[])` | Checks if two shapes match. |
| `ThrowIfNativeModeUnavailable` | Throws if the native-mode components (Generator, Discriminator, loss) have not been initialized. |
| `ThrowIfOnnxMode` | Throws if the model is running in ONNX mode where native operations are not supported. |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainStep(Tensor<>,Tensor<>)` | Performs one training step for Real-ESRGAN. |
| `UpdateParameters(Vector<>)` |  |
| `Upscale(Tensor<>)` | Upscales a low-resolution image to high resolution. |
| `ValidateAndGetArchitecture(NeuralNetworkArchitecture<>,NeuralNetworkArchitecture<>,InputType)` | Validates constructor arguments and throws if null (called before base constructor). |

## Fields

| Field | Summary |
|:-----|:--------|
| `_discriminatorOptimizer` | The optimizer used for training the discriminator network. |
| `_ganLambda` | Coefficient for the GAN (adversarial) loss. |
| `_generatorLosses` | Loss values from the last training step for monitoring. |
| `_generatorOptimizer` | The optimizer used for training the generator network. |
| `_l1Lambda` | Coefficient for the L1 (pixel-wise) loss. |
| `_lastDiscriminatorLoss` | Stores the last discriminator loss for diagnostics. |
| `_lastGeneratorLoss` | Stores the last generator loss for diagnostics. |
| `_numFeatures` | Number of feature channels. |
| `_numRRDBBlocks` | Number of RRDB blocks in the generator. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the generator model. |
| `_perceptualLambda` | Coefficient for the perceptual (VGG) loss. |
| `_realESRGANLoss` | The combined loss function for Real-ESRGAN training. |
| `_residualScale` | Residual scaling factor for training stability. |
| `_scaleFactor` | The upscaling factor (2, 4, or 8). |
| `_useNativeMode` | Indicates whether this Real-ESRGAN uses native layers (true) or ONNX model (false). |

