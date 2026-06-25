---
title: "ShapEModel<T>"
description: "Shap-E model for text-to-3D and image-to-3D generation with implicit neural representations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Shap-E model for text-to-3D and image-to-3D generation with implicit neural representations.

## For Beginners

Shap-E creates 3D objects that you can view from any angle:

What is an Implicit Neural Representation (NeRF)?

- A neural network that knows the 3D shape
- Input: 3D coordinates (x, y, z)
- Output: Color and density at that point
- Can render views from ANY angle without artifacts

Comparison with Point-E:
| Feature | Point-E | Shap-E |
|----------------|--------------|---------------|
| Output | Point cloud | Neural field |
| Quality | Good | Better |
| Rendering | Fast | Slower |
| Mesh export | Reconstruction | Direct SDF |
| Memory | Lower | Higher |

Example: "A red chair"

1. Shap-E generates network weights (latent representation)
2. These weights define a neural network
3. Query (x,y,z) -> neural network -> color, density
4. Render from any view or extract mesh via marching cubes

Use cases:

- High-quality 3D assets
- Novel view synthesis
- Direct mesh export with SDF
- View-consistent 3D models

## How It Works

Shap-E is OpenAI's model for generating 3D objects as implicit neural representations
(NeRFs). Unlike Point-E which generates point clouds, Shap-E generates parameters
for a neural network that represents the 3D shape, which can then be rendered
from any angle or converted to meshes.

Technical specifications:

- Latent dimension: 1024 parameters per shape
- Output: NeRF weights or SDF (Signed Distance Function)
- Rendering: Differentiable volumetric rendering
- Mesh export: Marching cubes on SDF
- Inference: ~64 steps

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShapEModel(NeuralNetworkArchitecture<>,DiffusionModelOptions<>,INoiseScheduler<>,DiTNoisePredictor<>,IConditioningModule<>,Boolean,Int32,Nullable<Int32>)` | Initializes a new Shap-E model with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Conditioner` |  |
| `LatentChannels` |  |
| `LatentDimension` | Gets the latent dimension. |
| `NoisePredictor` |  |
| `ParameterCount` |  |
| `SupportsMesh` |  |
| `SupportsNovelView` |  |
| `SupportsPointCloud` |  |
| `SupportsScoreDistillation` |  |
| `SupportsTexture` |  |
| `UseSDFMode` | Gets whether this model uses SDF mode. |
| `VAE` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CreateDummyVAE` | Creates a dummy VAE for interface compliance. |
| `DeepCopy` |  |
| `EvaluateSDF(ReadOnlySpan<>,Double[])` | Evaluates the signed distance function at a point. |
| `ExtractMesh(Tensor<>,Int32)` | Extracts a mesh from the latent using marching cubes. |
| `FlattenToCondition(Tensor<>)` | Flattens an image latent to a conditioning vector. |
| `GenerateLatent(String,String,Int32,Double,Nullable<Int32>)` | Generates a latent representation of a 3D shape from text. |
| `GenerateLatentFromImage(Tensor<>,Int32,Double,Nullable<Int32>)` | Generates a latent from an image. |
| `GenerateMesh(String,String,Int32,Int32,Double,Nullable<Int32>)` | Generates a mesh directly from a text prompt. |
| `GeneratePointCloud(String,String,Nullable<Int32>,Int32,Double,Nullable<Int32>)` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers(DiTNoisePredictor<>,Nullable<Int32>)` | Initializes the latent predictor and image VAE layers. |
| `QueryNeRF(ReadOnlySpan<>,Double[])` | Queries the NeRF at a 3D point (simplified). |
| `RenderView(Tensor<>,ValueTuple<Double,Double,Double>,ValueTuple<Double,Double,Double>,Int32,Int32)` | Renders a view of the shape from a camera position. |
| `SampleNeRFColor(ReadOnlySpan<>,Double[],Double[],Int32)` | Samples color along a ray using the NeRF representation. |
| `SamplePointCloud(Tensor<>,Nullable<Int32>,Nullable<Int32>)` | Samples a point cloud from the shape. |
| `SetParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `NERF_MLP_HIDDEN` | MLP hidden dimension. |
| `NERF_MLP_LAYERS` | Number of MLP layers for NeRF decoder. |
| `SHAPE_CONTEXT_DIM` | Context dimension for conditioning (CLIP embedding size). |
| `SHAPE_HIDDEN_SIZE` | Hidden size for the latent diffusion transformer. |
| `SHAPE_LATENT_DIM` | Standard Shap-E latent dimension. |
| `SHAPE_NUM_HEADS` | Number of attention heads. |
| `SHAPE_NUM_LAYERS` | Number of transformer layers. |
| `_conditioner` | The conditioning module (CLIP for text/image encoding). |
| `_imageVAE` | Standard VAE for image encoding. |
| `_latentPredictor` | The latent diffusion transformer. |
| `_useSDFMode` | Whether to generate SDF (Signed Distance Function) or NeRF. |

