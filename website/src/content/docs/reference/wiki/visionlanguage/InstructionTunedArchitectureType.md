---
title: "InstructionTunedArchitectureType"
description: "Specifies the architecture type for instruction-tuned vision-language models."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies the architecture type for instruction-tuned vision-language models.

## Fields

| Field | Summary |
|:-----|:--------|
| `CrossAttentionResampler` | Cross-attention resampler: vision encoder -> cross-attention resampler -> LLM (Qwen-VL series). |
| `DirectPatch` | Direct patch embedding: raw image patches go directly into the language model without a separate vision encoder (Fuyu). |
| `MLPProjection` | MLP projection: vision encoder -> MLP connector -> LLM (LLaVA, InternVL, DeepSeek-VL, Phi-3-Vision). |
| `QFormerProjection` | Q-Former projection: vision encoder -> Q-Former -> linear projection -> LLM (MiniGPT-4, MiniGPT-v2). |
| `VisualAbstractor` | Visual abstractor: vision encoder -> learnable visual abstractor module -> LLM (mPLUG-Owl series). |
| `VisualExpert` | Visual expert: vision encoder -> visual expert modules interleaved in every LLM layer (CogVLM). |

