---
title: "PointEModel<T>"
description: "Point-E model for text-to-3D point cloud generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Point-E model for text-to-3D point cloud generation.

## For Beginners

Point-E creates 3D objects as "point clouds" - collections
of colored points in 3D space that form a shape:

What is a point cloud?

- Thousands of 3D points (X, Y, Z coordinates)
- Each point can have a color (R, G, B)
- Together they form the surface of an object
- Like a very detailed dot-to-dot drawing in 3D

Example: "A red chair"

1. Point-E first imagines what the chair looks like (synthetic image)
2. Then generates 4096 points forming the chair shape
3. Points are colored red where appropriate
4. Result: A 3D point cloud you can view from any angle

Use cases:

- 3D modeling: Quick prototypes for games, VR, AR
- Visualization: Create 3D representations from descriptions
- Dataset creation: Generate synthetic 3D training data

## How It Works

Point-E is OpenAI's model for generating 3D point clouds from text descriptions.
It uses a two-stage pipeline: first generating a synthetic view of the object,
then generating a point cloud conditioned on that view.

Technical specifications:

- Default point count: 4096 (can generate 1024, 4096, or 16384)
- Coordinate range: [-1, 1] normalized
- Color: RGB values [0, 1]
- Two-stage: Image generation + point cloud diffusion
- Inference: ~40 steps for image, ~64 for point cloud

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PointEModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,ILatentDiffusionModel<>,IConditioningModule<>,Int32,Boolean,Nullable<Int32>)` | Initializes a new Point-E model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `PointCloudPredictor` | Gets the point cloud predictor. |
| `SupportsMesh` |  |
| `SupportsNovelView` |  |
| `SupportsPointCloud` |  |
| `SupportsScoreDistillation` |  |
| `SupportsTexture` |  |
| `UsesTwoStage` | Gets whether this model uses two-stage generation. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CombineConditions(Tensor<>,Tensor<>)` | Combines text and image conditions. |
| `ConvertToMesh(Tensor<>,Int32)` | Converts point cloud to mesh using marching cubes (simplified). |
| `CreateDummyVAE` | Creates a dummy VAE for interface compliance (Point-E uses point cloud directly). |
| `DeepCopy` |  |
| `EncodeImageCondition(Tensor<>)` | Encodes an image for conditioning. |
| `GenerateFromImage(Tensor<>,Nullable<Int32>,Int32,Nullable<Int32>)` | Generates a point cloud from an image. |
| `GeneratePointCloud(String,String,Nullable<Int32>,Int32,Double,Nullable<Int32>)` | Generates a colored point cloud from a text prompt. |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,Nullable<Int32>)` | Initializes the point cloud predictor and image VAE layers. |
| `NormalizePointCloud(Tensor<>)` | Normalizes point cloud coordinates and colors. |
| `PredictPointNoise(Tensor<>,Int32,Tensor<>)` | Predicts noise for point cloud denoising. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `POINTE_CONTEXT_DIM` | Context dimension for conditioning (CLIP embedding size). |
| `POINTE_HIDDEN_SIZE` | Default hidden size for the DiT transformer predictor. |
| `POINTE_LATENT_CHANNELS` | Standard Point-E latent channels. |
| `POINTE_NUM_HEADS` | Default number of attention heads. |
| `POINTE_NUM_LAYERS` | Default number of transformer layers. |
| `_conditioner` | The conditioning module (CLIP for text/image encoding). |
| `_imageGenerator` | The image generator for the first stage (optional). |
| `_imageVAE` | Standard VAE for image encoding (used in image-to-3D). |
| `_pointCloudPredictor` | The point cloud noise predictor (transformer-based). |
| `_useTwoStage` | Whether to use the two-stage pipeline. |

