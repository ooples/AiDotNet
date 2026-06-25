---
title: "FederatedAdapterOptions"
description: "Configuration options for federated parameter-efficient fine-tuning (PEFT)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated parameter-efficient fine-tuning (PEFT).

## For Beginners

When fine-tuning large models (like LLMs) in a federated setting,
sending all model parameters between clients and server is too expensive. PEFT adapters only
train and communicate a tiny fraction of parameters (often <1%), making federated LLM
fine-tuning practical.

## Properties

| Property | Summary |
|:-----|:--------|
| `AdapterDPNoiseMultiplier` | Gets or sets the DP noise multiplier for adapter-level privacy. |
| `AdapterLevelDP` | Gets or sets whether to apply adapter-level differential privacy. |
| `AdapterType` | Gets or sets the adapter type. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for prompt tuning. |
| `LayerInputDimension` | Gets or sets the layer input dimension for LoRA. |
| `LayerOutputDimension` | Gets or sets the layer output dimension for LoRA. |
| `LoRAAlpha` | Gets or sets the LoRA alpha scaling factor. |
| `LoRARank` | Gets or sets the LoRA rank. |
| `MaxHeterogeneousRank` | Gets or sets the maximum LoRA rank for heterogeneous LoRA. |
| `NumAdaptedLayers` | Gets or sets the number of layers to apply LoRA adapters to. |
| `NumPromptTokens` | Gets or sets the number of soft prompt tokens for prompt tuning. |

