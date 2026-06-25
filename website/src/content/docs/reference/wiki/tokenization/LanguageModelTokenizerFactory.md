---
title: "LanguageModelTokenizerFactory"
description: "Factory for creating tokenizers appropriate for different language model backbones."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Tokenization`

Factory for creating tokenizers appropriate for different language model backbones.

## For Beginners

Each language model was trained with a specific tokenizer.
Using the wrong tokenizer will produce garbage results. This factory creates
a basic tokenizer with the correct special tokens for each model type.

For production use, you should load the actual pretrained tokenizer from
HuggingFace using `AutoTokenizer`.

## How It Works

Different language models use different tokenization schemes:

- OPT, Chinchilla: GPT-style BPE tokenization
- Flan-T5: T5-style SentencePiece tokenization
- LLaMA, Vicuna, Mistral: LLaMA-style SentencePiece tokenization
- Phi, Qwen: GPT-style BPE with custom vocabulary

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateForBackbone(LanguageModelBackbone,IEnumerable<String>,Int32)` | Creates a tokenizer appropriate for the specified language model backbone. |
| `GetHuggingFaceModelName(LanguageModelBackbone)` | Gets the recommended HuggingFace model name for loading a pretrained tokenizer. |
| `GetSpecialTokens(LanguageModelBackbone)` | Gets the special tokens configuration for a language model backbone. |

