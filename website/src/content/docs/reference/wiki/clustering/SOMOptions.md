---
title: "SOMOptions<T>"
description: "Configuration options for Self-Organizing Maps (SOM)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Self-Organizing Maps (SOM).

## For Beginners

SOM creates a "map" of your data.

Imagine compressing a 3D world onto a 2D map:

- Nearby countries on the map should be nearby in reality
- The map preserves relationships even though it's lower dimensional

SOM does this for high-dimensional data:

- Creates a 2D grid of "neurons"
- Each neuron represents a prototype pattern
- Similar data points activate nearby neurons

Uses:

- Visualization of high-dimensional data
- Dimensionality reduction that preserves topology
- Finding natural groupings in data

## How It Works

Self-Organizing Maps are a type of neural network that produces a low-dimensional
(typically 2D) discretized representation of the input space. They preserve
topological properties of the input, meaning similar inputs map to nearby neurons.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SOMOptions` | Initializes SOMOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistanceMetric` | Gets or sets the distance metric for input space. |
| `GridHeight` | Gets or sets the height of the SOM grid. |
| `GridWidth` | Gets or sets the width of the SOM grid. |
| `InitialLearningRate` | Gets or sets the initial learning rate. |
| `InitialNeighborhoodRadius` | Gets or sets the initial neighborhood radius. |
| `NeighborhoodType` | Gets or sets the neighborhood function type. |
| `Topology` | Gets or sets the topology of the SOM grid. |

