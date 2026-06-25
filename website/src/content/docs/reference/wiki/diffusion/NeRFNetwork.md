---
title: "NeRFNetwork<T>"
description: "Neural Radiance Field network for 3D representation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.ThreeD`

Neural Radiance Field network for 3D representation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeRFNetwork(Int32,Int32,Nullable<Int32>)` | Initializes a new NeRF network. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total parameter count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetParameters` | Gets all network parameters. |
| `Initialize` | Initializes network weights. |
| `PositionalEncoding(Double[],Int32)` | Applies positional encoding to input values. |
| `QueryColor(Double,Double,Double,Double,Double,Double)` | Queries the color at a 3D point with viewing direction. |
| `QueryDensity(Double,Double,Double)` | Queries the density at a 3D point. |
| `Render(CameraPose,Int32)` | Renders an image from the NeRF at a given camera pose. |
| `SetParameters(Vector<>)` | Sets network parameters. |
| `UpdateParameters(Tensor<>,CameraPose,Double)` | Updates network parameters using gradient. |
| `VolumeRender(Double,Double,Double,Double,Double,Double)` | Performs volume rendering along a ray. |

