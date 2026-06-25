---
title: "GenerativeArchitectureType"
description: "Specifies the architecture type for generative vision-language models."
section: "API Reference"
---

`Enums` · `AiDotNet.VisionLanguage.Encoders`

Specifies the architecture type for generative vision-language models.

## Fields

| Field | Summary |
|:-----|:--------|
| `CausalMultimodal` | Causal multimodal: visual tokens embedded directly in causal language model (KOSMOS). |
| `EncoderDecoder` | Encoder-decoder: ViT encoder + autoregressive text decoder with cross-attention (GIT, CoCa, PaLI). |
| `PerceiverResampler` | Perceiver resampler: latent queries cross-attend to vision, gated cross-attention into LLM (Flamingo, IDEFICS). |
| `QFormerBridge` | Q-Former bridge: learnable queries cross-attend to vision, then feed decoder (InstructBLIP, BLIP-3). |
| `UnifiedGeneration` | Unified generation: single model for both understanding and image/text generation (Emu). |

