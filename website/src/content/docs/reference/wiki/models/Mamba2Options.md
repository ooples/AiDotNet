---
title: "Mamba2Options<T>"
description: "Configuration options for Mamba-2 (Structured State Space Duality) forecasting."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Mamba-2 (Structured State Space Duality) forecasting.

## For Beginners

Mamba-2 is an evolution of the Mamba architecture:

**Key Improvements over Mamba-1:**

1. **SSD Algorithm:** Uses matrix multiply instead of associative scan — much faster on GPUs
2. **Multi-head Structure:** Like multi-head attention, enabling better capacity per parameter
3. **Chunk-wise Processing:** Processes sequences in chunks for better hardware utilization
4. **2-8x Faster Training:** Due to better hardware mapping

**Architecture:**

- Input projection to expanded dimension
- Multi-head structured state space blocks
- Chunk-wise parallel processing
- Output projection back to model dimension

## How It Works

Mamba-2 improves upon Mamba by discovering the connection between selective state space models
and structured masked attention (State Space Duality). This enables a more efficient SSD algorithm
using matrix multiplications rather than associative scans, achieving 2-8x faster training.

**Reference:** Dao and Gu, "Transformers are SSMs: Generalized Models and Efficient Algorithms
Through Structured State Space Duality", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Mamba2Options` | Initializes a new instance with default values. |
| `Mamba2Options(Mamba2Options<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ChunkSize` | Gets or sets the chunk size for SSD computation. |
| `ContextLength` | Gets or sets the context length (input sequence length). |
| `ConvKernelSize` | Gets or sets the convolution kernel size. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ExpandFactor` | Gets or sets the expansion factor for the inner dimension. |
| `ForecastHorizon` | Gets or sets the forecast horizon (prediction length). |
| `ModelDimension` | Gets or sets the model dimension (d_model). |
| `NumHeads` | Gets or sets the number of heads for multi-head SSD. |
| `NumLayers` | Gets or sets the number of Mamba-2 layers. |
| `StateDimension` | Gets or sets the state dimension per head. |

