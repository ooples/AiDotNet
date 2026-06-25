---
title: "Ray<T>"
description: "Represents a ray in 3D space for rendering."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralRadianceFields.Data`

Represents a ray in 3D space for rendering.

## How It Works

**For Beginners:** A ray is a half-line starting from a point and extending in a direction.

In computer graphics and rendering:

- Rays are cast from camera through each pixel
- They represent the path light travels
- We sample points along rays to determine what we see

Ray equation:

- Point on ray = Origin + t * Direction
- t is the distance along the ray (t >= 0)
- Origin: Starting point (usually camera position)
- Direction: Which way the ray points (unit vector)

Example:

- Camera at origin (0, 0, 0)
- Looking down negative Z axis
- Ray for center pixel: Origin = (0, 0, 0), Direction = (0, 0, -1)
- Point at distance 5: (0, 0, 0) + 5 * (0, 0, -1) = (0, 0, -5)

In NeRF:

- Each pixel corresponds to one ray
- We sample many points along each ray
- Query the neural network at each sample point
- Blend results to get final pixel color

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Ray(Vector<>,Vector<>,,)` | Initializes a new instance of the Ray class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Direction` | Gets or sets the direction vector of the ray (should be normalized). |
| `FarBound` | Gets or sets the far bound for sampling along the ray. |
| `NearBound` | Gets or sets the near bound for sampling along the ray. |
| `Origin` | Gets or sets the origin point of the ray. |

## Methods

| Method | Summary |
|:-----|:--------|
| `PointAt()` | Computes a point along the ray at a specific distance. |

