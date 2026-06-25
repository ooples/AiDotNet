---
title: "LanguageModelBackbone"
description: "Defines the language model backbone types used in multimodal neural networks."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the language model backbone types used in multimodal neural networks.

## For Beginners

Think of the language model backbone as the "brain" that processes
and generates text in vision-language models.

When a model like BLIP-2 needs to describe an image or answer a question about it:

1. The vision encoder extracts features from the image
2. The Q-Former/adapter bridges vision and language
3. The language model backbone generates the actual text response

Different backbones have different strengths:

- **OPT**: Good for general text generation, used in BLIP-2
- **FlanT5**: Better for instruction-following, used in BLIP-2
- **LLaMA**: Efficient and powerful, used in LLaVA
- **Vicuna**: LLaMA fine-tuned for conversations, used in LLaVA
- **Mistral**: Fast and efficient, newer alternative for LLaVA
- **Chinchilla**: Used in Flamingo, optimized for multimodal learning

The choice affects model size, speed, and quality of text generation.

## How It Works

This enum specifies which language model architecture is used as the backbone for text generation
and understanding in multimodal models like BLIP-2, LLaVA, and Flamingo. The backbone determines
the model's capacity, vocabulary, and generation capabilities.

## Fields

| Field | Summary |
|:-----|:--------|
| `Chinchilla` | Chinchilla by DeepMind - compute-optimal language model. |
| `FlanT5` | Flan-T5 by Google - instruction-tuned T5 model. |
| `LLaMA` | LLaMA (Large Language Model Meta AI) by Meta. |
| `Mistral` | Mistral - efficient open-source language model. |
| `OPT` | OPT (Open Pre-trained Transformer) by Meta AI. |
| `Phi` | Phi by Microsoft - small but capable language model. |
| `Qwen` | Qwen by Alibaba - multilingual language model. |
| `RoBERTa` | RoBERTa - robustly optimized BERT-style encoder. |
| `Vicuna` | Vicuna - LLaMA fine-tuned on conversational data. |

