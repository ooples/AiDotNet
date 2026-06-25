---
title: "CogVLM<T>"
description: "CogVLM: deep fusion via visual expert module in every LLM layer."
section: "API Reference"
---

`Models & Types` · `AiDotNet.VisionLanguage.InstructionTuned`

CogVLM: deep fusion via visual expert module in every LLM layer.

## For Beginners

CogVLM introduces a clever "visual expert" approach — instead
of just prepending image tokens to the text, it adds a dedicated visual expert module inside
every layer of the language model. Each decoder layer has separate attention weights and
feed-forward weights for visual tokens versus text tokens. This deep fusion means the model
can align visual and language features much more precisely without degrading the original
language model's text capabilities. It uses a massive EVA2-CLIP-E vision encoder (4.4B
parameters, 63 layers) and Vicuna as the decoder. Default values follow the original paper
settings.

## How It Works

CogVLM (Wang et al., 2023) introduces a visual expert module in every layer of the language
model for deep fusion of visual and language features. Each decoder layer has separate QKV
matrices and FFN weights for visual tokens (the visual expert), enabling fine-grained
visual-linguistic alignment without degrading language model capability.

**References:**

- Paper: "CogVLM: Visual Expert for Pretrained Language Models" (Wang et al., 2023)

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateFromImage(Tensor<>,String)` | Generates text using CogVLM's visual expert module architecture. |

