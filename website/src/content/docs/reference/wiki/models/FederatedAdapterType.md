---
title: "FederatedAdapterType"
description: "Specifies the federated adapter type for parameter-efficient fine-tuning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the federated adapter type for parameter-efficient fine-tuning.

## Fields

| Field | Summary |
|:-----|:--------|
| `DPFedLoRA` | DP-FedLoRA — differentially private LoRA with per-layer noise calibration. |
| `FLoRA` | FLoRA — stacked lossless LoRA aggregation via SVD. |
| `FedAdapter` | FedAdapter — bottleneck adapter layers inserted into transformer blocks. |
| `FedMeZO` | FedMeZO — memory-efficient zeroth-order optimization for LLM fine-tuning. |
| `FedPETuning` | FedPETuning — unified PEFT framework (LoRA, adapters, prefix, BitFit). |
| `HeterogeneousLoRA` | Heterogeneous LoRA — different ranks per client with SVD aggregation. |
| `HierFedLoRA` | HierFedLoRA — hierarchical LoRA for edge-cloud topologies. |
| `LoRA` | LoRA — Low-Rank Adaptation with uniform rank across clients. |
| `None` | No adapter — standard full-model aggregation. |
| `PromptTuning` | Prompt Tuning — soft prompt token aggregation. |
| `SLoRA` | SLoRA — sparse LoRA that only communicates non-zero adapter elements. |

