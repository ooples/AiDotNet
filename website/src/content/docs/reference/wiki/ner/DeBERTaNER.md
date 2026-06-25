---
title: "DeBERTaNER<T>"
description: "DeBERTa-NER: Decoding-enhanced BERT with disentangled Attention for NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

DeBERTa-NER: Decoding-enhanced BERT with disentangled Attention for NER.

## For Beginners

DeBERTa is one of the most accurate transformer models for NER.
It improves on BERT by being smarter about how it handles word positions. In normal BERT,
the word's meaning and its position in the sentence are mixed together. DeBERTa keeps them
separate, which helps it better understand relationships between words.

Use DeBERTa-NER when you want the highest possible accuracy and can afford the compute cost.

## How It Works

DeBERTa-NER (He et al., ICLR 2021 - "DeBERTa: Decoding-enhanced BERT with Disentangled Attention")
is a state-of-the-art transformer model that introduces two key architectural innovations:

**1. Disentangled Attention:**
Instead of combining content and position into a single embedding (as in BERT), DeBERTa
represents each token with two separate vectors: one for content and one for position.
The attention score between two tokens is computed as the sum of four components:

- Content-to-content: semantic similarity
- Content-to-position: how important a token's meaning is relative to another's position
- Position-to-content: how important a token's position is relative to another's meaning
- Position-to-position: relative position bias

This disentangled approach captures richer token relationships, particularly beneficial for
NER where both semantic meaning and positional context matter.

**2. Enhanced Mask Decoder:**
Incorporates absolute position information in the final decoding layers, combining the
benefits of relative position (used throughout the model) with absolute position (needed
for tasks sensitive to word order).

**Performance (CoNLL-2003):**

- DeBERTa-base: ~93.1% F1
- DeBERTa-large: ~93.5% F1 (state-of-the-art for single models)
- DeBERTa-xlarge: ~93.8% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeBERTaNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a DeBERTa-NER model in ONNX inference mode. |
| `DeBERTaNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a DeBERTa-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

