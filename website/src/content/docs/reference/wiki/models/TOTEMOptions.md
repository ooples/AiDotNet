---
title: "TOTEMOptions<T>"
description: "Configuration options for TOTEM (TOkenized Time Series EMbeddings)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TOTEM (TOkenized Time Series EMbeddings).

## For Beginners

TOTEM bridges continuous time series and discrete tokens:

**Vector Quantization (VQ-VAE):**
TOTEM maintains a learned codebook of discrete "patterns". Each segment of your time
series is matched to the nearest codebook entry, converting continuous values into
discrete tokens. This allows LLM-style methods to work on time series.

**Key Advantages:**

- Converts continuous time series to discrete tokens for LLM compatibility
- Multiple codebooks capture different aspects of temporal patterns
- Commitment loss keeps encoder outputs close to codebook entries

## How It Works

TOTEM learns discrete tokenized representations for time series via VQ-VAE,
enabling the use of discrete token-based methods (like LLMs) on continuous time series data.

**Reference:** Talukder et al., "TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis", 2024.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TOTEMOptions` | Initializes a new instance with default values. |
| `TOTEMOptions(TOTEMOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CodebookDimension` | Gets or sets the dimension of each codebook entry. |
| `CodebookSize` | Gets or sets the number of entries in each codebook. |
| `CommitmentWeight` | Gets or sets the commitment loss weight for VQ training. |
| `ContextLength` | Gets or sets the number of historical time steps used as input context. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `ForecastHorizon` | Gets or sets the number of future time steps to forecast. |
| `HiddenDimension` | Gets or sets the hidden dimension of the transformer layers. |
| `NumCodebooks` | Gets or sets the number of parallel codebooks (product quantization). |
| `NumHeads` | Gets or sets the number of attention heads. |
| `NumLayers` | Gets or sets the number of transformer layers. |

