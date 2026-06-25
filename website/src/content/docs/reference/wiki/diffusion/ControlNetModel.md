---
title: "ControlNetModel<T>"
description: "ControlNet model for adding spatial conditioning to diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Control`

ControlNet model for adding spatial conditioning to diffusion models.

## For Beginners

ControlNet is like giving the AI artist a reference sketch
or blueprint to follow while creating an image.

Supported control types:

- Canny edges: Outline/edge detection of shapes
- Depth maps: 3D depth information
- Pose keypoints: Human body positions (OpenPose)
- Segmentation: Region/object boundaries
- Normal maps: Surface orientation
- Scribbles: Simple user drawings
- Line art: Clean line drawings

How it works:

1. You provide a control image (e.g., edge map of a house)
2. ControlNet encodes this control signal
3. The encoded control guides the diffusion process
4. Result: Generated image follows the control structure

Example use cases:

- "Draw a Victorian house" + edge map = house in exact shape
- "Dancing woman" + pose skeleton = person in exact pose
- "Forest scene" + depth map = correct 3D perspective

## How It Works

ControlNet enables fine-grained spatial control over image generation by adding
additional conditioning signals such as edge maps, depth maps, pose keypoints,
segmentation masks, and more. It works by creating a trainable copy of the
encoder blocks that process the control signal.

Technical details:

- ControlNet is a "zero convolution" architecture
- Copies encoder weights from base model
- Adds control signal via residual connections
- Can be combined: multi-ControlNet stacking
- Supports conditioning strength adjustment (0-1)

Reference: Zhang et al., "Adding Conditional Control to Text-to-Image Diffusion Models", 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ControlNetModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,UNetNoisePredictor<>,StandardVAE<>,IConditioningModule<>,ControlType,Nullable<Int32>)` | Initializes a new instance of ControlNetModel with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `ConditioningStrength` | Gets or sets the default conditioning strength (0-1). |
| `ControlType` | Gets the type of control signal this model uses. |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddTensors(Tensor<>,Tensor<>)` | Adds two tensors element-wise with proper shape handling. |
| `Clone` |  |
| `DeepCopy` |  |
| `GenerateWithControl(String,Tensor<>,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Double>,Nullable<Int32>)` | Generates an image with control signal. |
| `GenerateWithControlFeatures(String,List<Tensor<>>,String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image using pre-computed control features. |
| `GenerateWithMultiControl(String,Tensor<>[],ControlType[],Double[],String,Int32,Int32,Int32,Nullable<Double>,Nullable<Int32>)` | Generates an image with multiple control signals. |
| `GetControlChannels(ControlType)` | Gets the number of input channels for a control type. |
| `GetModelMetadata` |  |
| `GetOrCreateEncoder(ControlType)` | Gets or creates a cached encoder for the specified control type. |
| `GetParameters` |  |
| `InitializeLayers(UNetNoisePredictor<>,StandardVAE<>,Nullable<Int32>)` | Initializes the U-Net, VAE, and ControlNet encoder layers. |
| `PredictWithControl(Tensor<>,Int32,Tensor<>,List<Tensor<>>)` | Predicts noise with control signal integration. |
| `ScaleTensor(Tensor<>,Double)` | Scales a tensor by a scalar value. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `CN_LATENT_CHANNELS` | Standard ControlNet latent channels. |
| `CN_VAE_SCALE_FACTOR` | Standard VAE scale factor. |
| `_baseUNet` | The base noise predictor (U-Net). |
| `_conditioner` | The conditioning module for text encoding. |
| `_conditioningStrength` | Default conditioning strength. |
| `_controlChannels` | Number of input channels for control signal. |
| `_controlNetEncoder` | The ControlNet encoder blocks for the primary control type. |
| `_controlType` | The type of control signal this model handles. |
| `_encoderCache` | Cache of encoders by control type for multi-control generation. |
| `_vae` | The VAE for encoding/decoding. |

