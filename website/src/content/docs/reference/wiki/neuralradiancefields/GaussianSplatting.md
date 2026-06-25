---
title: "GaussianSplatting<T>"
description: "Implements 3D Gaussian Splatting for real-time novel view synthesis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralRadianceFields.Models`

Implements 3D Gaussian Splatting for real-time novel view synthesis.

## For Beginners

3D Gaussian Splatting is the newest breakthrough in novel view synthesis,
achieving real-time rendering (30+ FPS) with state-of-the-art quality.

## How It Works

Speed comparison:

- NeRF: ~30 seconds per frame
- Instant-NGP: ~30 milliseconds per frame (~1000× faster than NeRF)
- Gaussian Splatting: ~10 milliseconds per frame (~3000× faster than NeRF)
- Gaussian Splatting can render at 100+ FPS on modern GPUs!

Key innovation - Explicit representation:

- NeRF/Instant-NGP: Implicit (neural network represents the scene)
- Gaussian Splatting: Explicit (scene is a collection of 3D Gaussians)

What's a 3D Gaussian?

- A "blob" of color in 3D space
- Has a center position (x, y, z)
- Has a size and shape (covariance matrix)
- Has a color (RGB)
- Has opacity (alpha)

Think of it like this:

- Traditional rendering: Triangles (hard edges)
- Volume rendering (NeRF): Sample continuous field
- Gaussian Splatting: "Paint" with fuzzy 3D blobs

How it works:

1. Start with point cloud (from SfM like COLMAP)
2. Place a Gaussian at each point
3. Optimize Gaussian parameters:
- Position: Where is the Gaussian?
- Covariance: What shape/size is it?
- Color: What color is it?
- Opacity: How transparent is it?
4. Add/remove Gaussians as needed (adaptive densification)
5. Render by "splatting" Gaussians onto image

Gaussian Splatting rendering:

1. For each Gaussian:
- Project to 2D image plane
- Compute 2D Gaussian on screen
- Determines which pixels it affects
2. Sort Gaussians by depth (back to front)
3. For each pixel:
- Blend Gaussians that affect it (alpha blending)
- Front-to-back or back-to-front blending
4. Result: Final pixel color

This is MUCH faster than ray marching because:

- No network evaluation (explicit representation)
- Highly parallelizable (each Gaussian independent)
- Efficient GPU rasterization (like traditional graphics)

Why Gaussians?

- Smooth gradients for optimization
- Can be rasterized efficiently (like triangles)
- 2D projection is also Gaussian (mathematically elegant)
- Adaptive: Can represent sharp edges with many small Gaussians

or smooth surfaces with fewer large Gaussians

Gaussian parameters:

- Position μ: Center of Gaussian (3D point)
- Covariance Σ: Shape and orientation (3×3 matrix)
- Can represent ellipsoids (stretched in different directions)
- Encoded as rotation + scale for easier optimization
- Color c: RGB values (often with spherical harmonics for view-dependent effects)
- Opacity α: Transparency (0 = invisible, 1 = opaque)

Total per Gaussian: 3 (pos) + 4 (rotation) + 3 (scale) + 3 (color) + 1 (opacity) = 14 values
With spherical harmonics: Can be 40-60 values per Gaussian

Adaptive densification:
During optimization, Gaussians are dynamically added/removed:

- Clone: Copy Gaussians in high-gradient regions (need more detail)
- Split: Large Gaussians → multiple smaller ones (refine detail)
- Prune: Remove transparent Gaussians (save memory)

Example:

- Start: 100K Gaussians from point cloud
- After optimization: 500K-5M Gaussians
- High detail areas: Many small Gaussians (e.g., edges, textures)
- Smooth areas: Fewer large Gaussians (e.g., walls, sky)

Training process:

1. Initialize from Structure-from-Motion point cloud
2. For each training iteration:
- Render view using current Gaussians
- Compute loss (difference from ground truth image)
- Backpropagate to Gaussian parameters
- Update positions, colors, shapes, opacities
- Every N iterations: Densify/prune Gaussians
3. Converges in ~10-30 minutes (similar to Instant-NGP)

Advantages over NeRF:

- Faster rendering: 100+ FPS vs 0.03 FPS (NeRF)
- Explicit representation: Easy to edit, manipulate
- No network evaluation: Simpler deployment
- Better quality: Often sharper details
- Easier to understand: Physical interpretation (colored blobs)

Disadvantages:

- Memory: More than NeRF (millions of Gaussians)
- NeRF: ~5-50MB
- Gaussian Splatting: ~100-500MB
- Requires good initialization (SfM point cloud)
- Can have "floating" artifacts in empty regions
- File size for storage

Rendering pipeline (simplified):
```
For each view:

1. Transform Gaussians to camera space
2. Project 3D Gaussians to 2D screen space
3. Compute 2D Gaussian parameters (center, covariance)
4. Determine which tiles/pixels each Gaussian affects
5. Sort Gaussians by depth within each tile
6. For each pixel:
- Accumulate color from affecting Gaussians
- Alpha blending: C = Σ α_i * c_i * Π(1 - α_j) for j < i
7. Output: Rendered image

```

Applications (especially suited for):

- Real-time VR/AR: Low latency is critical
- Gaming: Integration with game engines
- Digital twins: Interactive 3D models of real places
- Telepresence: Realistic remote environments
- Film pre-visualization: Fast preview of captured scenes
- Live events: Real-time volumetric capture

Comparison table:

Feature | NeRF | Instant-NGP | Gaussian Splatting
---------------------|---------|-------------|-------------------
Rendering speed | 30 s | 30 ms | 10 ms (100+ FPS)
Training time | 1-2 days| 5-10 min | 10-30 min
Quality | High | High | Very High
Memory usage | 5 MB | 50 MB | 200-500 MB
Editability | Hard | Hard | Easy
Real-time rendering | No | Borderline | Yes
GPU requirement | Any | CUDA | Any (faster w/ CUDA)

Reference: "3D Gaussian Splatting for Real-Time Radiance Field Rendering"
by Kerbl et al., SIGGRAPH 2023

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianSplatting(Matrix<>,Matrix<>,Boolean,Int32,ILossFunction<>)` | Initializes a new instance of the GaussianSplatting class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `GaussianCount` | Gets the number of Gaussians currently in the scene. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignRayTargetToPrediction(Tensor<>,Tensor<>)` | Reshape / channel-pad/truncate the target so it can be loss-compared element-wise to the prediction. |
| `ApplyRayGradients(Tensor<>,Tensor<>)` | Backprop helper for `Tensor{`: distributes ray-level colour gradients onto the Gaussian colour parameters. |
| `GetOptions` |  |
| `GetParameterChunks` | Yields the Gaussian state as a single chunk for parameter-diff iteration. |
| `GetParameters` | Flattens every Gaussian's trainable state (position, rotation, scale, opacity, colour) into a single contiguous vector. |
| `SeedDefaultGaussianCloud` | Seeds a small unit-cube Gaussian cloud (up to 8 corners) so the parameterless constructor produces a model with non-empty trainable state. |
| `SeedPlaceholderGaussianCloud(Int32)` | Seeds a Gaussian cloud with `count` placeholder points at the origin. |
| `TrainOnRays(Tensor<>,Tensor<>)` | Ray-level training path: input is [N, 6] (position + direction per ray), expectedOutput is [N, 3] ray colors. |

