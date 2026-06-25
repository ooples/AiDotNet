---
title: "ScanPatterns<T>"
description: "Provides scanning pattern functions for Vision SSM architectures that process 2D spatial data."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Provides scanning pattern functions for Vision SSM architectures that process 2D spatial data.

## For Beginners

When using Mamba for images instead of text, we need to turn a 2D grid of
image patches into a 1D sequence. The order we read the patches matters a lot!

Think of reading a page:

- Left-to-right, top-to-bottom (normal reading) = basic raster scan
- Reading both forward and backward = bidirectional scan (catches more patterns)
- Reading in all four directions = cross-scan (captures all spatial relationships)
- Reading in a zigzag pattern = continuous scan (keeps nearby patches close in sequence)

Each pattern captures different spatial relationships and works better for different tasks.

## How It Works

Vision Mamba models need to convert 2D patch grids into 1D sequences for SSM processing.
Different scanning patterns capture different spatial relationships:

## Methods

| Method | Summary |
|:-----|:--------|
| `BidirectionalScan(Tensor<>)` | Creates a bidirectional scan by concatenating forward and reversed sequences. |
| `ContinuousScan(Tensor<>,Int32,Int32)` | Creates a continuous (serpentine/zigzag) scan that preserves spatial locality. |
| `CrossScan(Tensor<>,Int32,Int32)` | Creates four directional scans from a 2D patch grid for cross-scanning. |
| `MergeScanOutputs(List<Tensor<>>)` | Merges multiple scan outputs by averaging them element-wise. |
| `SpatioTemporalScan(Tensor<>,Int32,Int32,Int32)` | Creates spatio-temporal scans for video data, scanning spatially within each frame and temporally across frames. |

