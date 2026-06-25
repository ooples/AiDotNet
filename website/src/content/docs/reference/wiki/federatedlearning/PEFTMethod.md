---
title: "PEFTMethod"
description: "Specifies the PEFT method used by FedPETuning."
section: "API Reference"
---

`Enums` · `AiDotNet.FederatedLearning.Adapters`

Specifies the PEFT method used by FedPETuning.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adapter` | Adapter — Bottleneck adapter layers inserted into transformer blocks. |
| `BitFit` | BitFit — Only bias terms are trainable. |
| `LoRA` | LoRA — Low-Rank Adaptation matrices. |
| `PrefixTuning` | Prefix — Learnable prefix tokens prepended to each layer's key/value. |

