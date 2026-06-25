---
title: "PretrainedTokenizerModel"
description: "Specifies pretrained tokenizer models available from HuggingFace Hub."
section: "API Reference"
---

`Enums` · `AiDotNet.Tokenization.Configuration`

Specifies pretrained tokenizer models available from HuggingFace Hub.

## How It Works

**For Beginners:** These are industry-standard tokenizers that have been trained
on large text corpora. Each is designed for different use cases:

- BERT models: Best for understanding text (classification, Q&A, NER)
- GPT models: Best for text generation
- RoBERTa: Improved BERT with better training
- T5: Versatile text-to-text model
- DistilBERT: Faster, smaller BERT

## Fields

| Field | Summary |
|:-----|:--------|
| `AlbertBaseV2` | ALBERT Base v2 - A Lite BERT with parameter sharing. |
| `BertBaseCased` | BERT Base Cased - Preserves case information. |
| `BertBaseUncased` | BERT Base Uncased - The default choice for most NLP tasks. |
| `BertLargeCased` | BERT Large Cased - Large model preserving case. |
| `BertLargeUncased` | BERT Large Uncased - Larger model with better accuracy. |
| `CodeBertBase` | CodeBERT Base - BERT for programming languages. |
| `DistilBertBaseCased` | DistilBERT Base Cased - Distilled BERT preserving case. |
| `DistilBertBaseUncased` | DistilBERT Base Uncased - Distilled BERT (40% smaller, 60% faster). |
| `ElectraBase` | Electra Base - Efficient pretraining approach (base size). |
| `ElectraSmall` | Electra Small - Efficient pretraining approach. |
| `Gpt2` | GPT-2 - OpenAI's text generation model. |
| `Gpt2Large` | GPT-2 Large - Even larger GPT-2 variant. |
| `Gpt2Medium` | GPT-2 Medium - Larger GPT-2 variant. |
| `GraphCodeBert` | GraphCodeBERT - Code model with data flow. |
| `MicrosoftCodeBert` | Microsoft CodeBERT - Multi-language code model. |
| `RobertaBase` | RoBERTa Base - Robustly optimized BERT. |
| `RobertaLarge` | RoBERTa Large - Large RoBERTa model. |
| `T5Base` | T5 Base - Text-to-Text Transfer Transformer (base). |
| `T5Large` | T5 Large - Text-to-Text Transfer Transformer (large). |
| `T5Small` | T5 Small - Text-to-Text Transfer Transformer (small). |
| `XlnetBaseCased` | XLNet Base Cased - Autoregressive pretraining. |

