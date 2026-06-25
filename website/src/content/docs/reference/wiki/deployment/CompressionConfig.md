---
title: "CompressionConfig"
description: "Configuration for model compression - reducing model size while preserving accuracy."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Deployment.Configuration`

Configuration for model compression - reducing model size while preserving accuracy.

## For Beginners

Model compression makes your trained AI model smaller and faster to load.
Think of it like compressing a ZIP file - you get a smaller file that can be restored to its original form.

Why use compression?

- Smaller model files (50-90% size reduction)
- Faster model loading and deployment
- Lower storage and bandwidth costs
- Enables deployment on resource-constrained devices

Trade-offs:

- Some compression types are lossy (slight accuracy reduction, typically 1-2%)
- Compression/decompression adds a small processing overhead

Compression happens automatically when you save (serialize) a model and
decompression happens automatically when you load (deserialize) it.
You never need to handle compression manually.

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxAccuracyLossPercent` | Gets or sets the maximum acceptable accuracy loss percentage (default: 2.0). |
| `MaxIterations` | Gets or sets the maximum iterations for clustering algorithms (default: 100). |
| `Mode` | Gets or sets the compression mode (default: Automatic). |
| `NumClusters` | Gets or sets the number of clusters for weight clustering (default: 256). |
| `Precision` | Gets or sets the decimal precision for Huffman encoding (default: 4). |
| `RandomSeed` | Gets or sets the random seed for reproducible compression (default: null for random). |
| `Tolerance` | Gets or sets the convergence tolerance for clustering algorithms (default: 1e-6). |
| `Type` | Gets or sets the compression algorithm type (default: WeightClustering). |

