---
title: "ELECTRANER<T>"
description: "ELECTRA-NER: Efficiently Learning an Encoder that Classifies Token Replacements Accurately for NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

ELECTRA-NER: Efficiently Learning an Encoder that Classifies Token Replacements Accurately for NER.

## For Beginners

ELECTRA is a smart, efficient transformer that learns faster than BERT.
While BERT learns by filling in blank words (like a fill-in-the-blank test), ELECTRA learns
by detecting fake words (like a fact-checker). This is more efficient because ELECTRA learns
from every word in every sentence, not just the few blanked-out words.

Use ELECTRA-NER when:

- You want good accuracy with limited compute budget
- You need a smaller model that still performs well
- Training efficiency is important

## How It Works

ELECTRA-NER (Clark et al., ICLR 2020 - "ELECTRA: Pre-training Text Encoders as Discriminators
Rather Than Generators") uses a novel replaced token detection (RTD) pre-training objective
instead of masked language modeling. ELECTRA is particularly efficient because:

**Key Innovation - Replaced Token Detection:**
Instead of masking 15% of tokens and predicting them (like BERT), ELECTRA:

1. A small "generator" network (like a tiny BERT) predicts masked tokens
2. Some predicted tokens are plausible but wrong replacements
3. The main "discriminator" network learns to detect which tokens were replaced
4. Every token position provides a training signal (vs only 15% for BERT)

This means ELECTRA learns from ALL tokens in every sentence, making it 4x more efficient
than BERT at the same compute budget. An ELECTRA-small model matches BERT-base performance
using only 1/4 of the compute.

**Performance (CoNLL-2003):**

- ELECTRA-small: ~91.5% F1 (matching BERT-base with 1/4 compute)
- ELECTRA-base: ~92.6% F1
- ELECTRA-large: ~93.3% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ELECTRANER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates an ELECTRA-NER model in ONNX inference mode. |
| `ELECTRANER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates an ELECTRA-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

