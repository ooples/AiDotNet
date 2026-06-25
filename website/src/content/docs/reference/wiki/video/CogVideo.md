---
title: "CogVideo<T>"
description: "CogVideo: Text-to-Video Diffusion Model for generating videos from text descriptions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Video.Generation`

CogVideo: Text-to-Video Diffusion Model for generating videos from text descriptions.

## For Beginners

CogVideo creates videos from text descriptions.
You provide a prompt like "a cat playing with a ball" and it generates
a video showing that scene. It works by:

1. Starting with random noise
2. Gradually denoising to create coherent frames
3. Ensuring temporal consistency across frames

Example usage (native mode for training):

Example usage (ONNX mode for inference only):

## How It Works

CogVideo is a state-of-the-art text-to-video generation model that:

- Generates coherent video clips from text prompts
- Uses diffusion-based denoising in latent space
- Produces temporally consistent animations

**Reference:** "CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer"
https://arxiv.org/abs/2408.06072

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CogVideo(NeuralNetworkArchitecture<>,String,Int32,Int32,CogVideoOptions)` | Creates a CogVideo model using a pretrained ONNX model for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmbedDim` | Gets the embedding dimension. |
| `NumFrames` | Gets the number of frames generated. |
| `SupportsTraining` | Gets whether training is supported (only in native mode). |
| `UseNativeMode` | Gets whether this model uses native mode (true) or ONNX mode (false). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineWithCondition(Tensor<>,Tensor<>,Tensor<>)` | Combines input with text and timestep conditioning. |
| `CreateNewInstance` |  |
| `CreateTimestepEmbedding(Double)` | Creates a timestep embedding using sinusoidal encoding. |
| `Denoise(Tensor<>,Tensor<>,Double)` | Performs a single denoising step. |
| `DenoisingStep(Tensor<>,Tensor<>,Double)` | Performs a single DDPM denoising step following Ho et al. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Forward(Tensor<>)` | Performs a forward pass through the network. |
| `Generate(Tensor<>,Int32)` | Generates video frames from a text embedding. |
| `GenerateRandomNoise` | Generates random noise for the diffusion process. |
| `GetModelMetadata` |  |
| `GetOptions` |  |
| `InitializeLayers` |  |
| `InitializeNoiseSchedule` | Initializes the DDPM noise schedule using a linear beta schedule. |
| `PredictCore(Tensor<>)` |  |
| `PredictNoise(Tensor<>,Tensor<>,Double)` | Predicts the noise component in the input. |
| `PredictOnnx(Tensor<>)` | Performs inference using the ONNX model. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultResolution` | Creates a CogVideo model using native layers for training and inference. |
| `_alphaBars` | Cumulative product of alphas (alpha_bar). |
| `_alphaBarsPrev` | Cumulative product of alphas for t-1 (alpha_bar_prev). |
| `_alphas` | Alpha values (1 - beta). |
| `_betas` | Beta schedule (variance at each timestep). |
| `_embedDim` | Embedding dimension for the model. |
| `_latentChannels` | Number of latent channels. |
| `_latentHeight` | Latent space height. |
| `_latentWidth` | Latent space width. |
| `_lossFunction` | The loss function for training. |
| `_numFrames` | Number of video frames to generate. |
| `_numLayers` | Number of transformer layers. |
| `_numTimesteps` | Number of diffusion timesteps. |
| `_onnxModelPath` | Path to the ONNX model file. |
| `_onnxSession` | The ONNX inference session for the model. |
| `_optimizer` | The optimizer used for training. |
| `_posteriorMeanCoef1` | Posterior mean coefficient 1: sqrt(alpha_bar_prev) * beta / (1 - alpha_bar). |
| `_posteriorMeanCoef2` | Posterior mean coefficient 2: sqrt(alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar). |
| `_posteriorVariances` | Posterior variance for DDPM sampling. |
| `_sqrtAlphaBars` | Square root of alpha_bar. |
| `_sqrtOneMinusAlphaBars` | Square root of 1 - alpha_bar. |
| `_useNativeMode` | Indicates whether this model uses native layers (true) or ONNX model (false). |

