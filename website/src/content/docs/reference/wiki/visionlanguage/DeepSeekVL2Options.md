---
title: "DeepSeekVL2Options"
description: "Options for DeepSeek-VL2 (MoE, dynamic tiling, multi-head latent attention)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.VisionLanguage.InstructionTuned`

Options for DeepSeek-VL2 (MoE, dynamic tiling, multi-head latent attention).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepSeekVL2Options(DeepSeekVL2Options)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableDynamicTiling` | Gets or sets whether dynamic tiling is enabled. |
| `NumActiveExperts` | Gets or sets the number of active MoE experts per token. |
| `NumExperts` | Gets or sets the number of MoE experts. |

