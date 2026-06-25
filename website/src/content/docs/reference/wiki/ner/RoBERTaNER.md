---
title: "RoBERTaNER<T>"
description: "RoBERTa-NER: Robustly Optimized BERT Approach with token classification for NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

RoBERTa-NER: Robustly Optimized BERT Approach with token classification for NER.

## For Beginners

RoBERTa is a "better-trained BERT." The architecture is identical,
but RoBERTa was trained more carefully on more data, resulting in consistently better performance.
Use RoBERTa-NER when you want the best accuracy from a standard transformer model.

## How It Works

RoBERTa-NER (Liu et al., 2019 - "RoBERTa: A Robustly Optimized BERT Pretraining Approach")
fine-tunes a RoBERTa model for NER. RoBERTa improves upon BERT through better pre-training:

**Key Improvements over BERT:**

- **No Next Sentence Prediction (NSP):** Removes the NSP objective which was found unhelpful
- **Dynamic masking:** Different tokens are masked in each epoch (vs BERT's static masking)
- **Longer training:** Trained on 160GB of text (10x more than BERT)
- **Larger batch sizes:** 8K sequences per batch for more stable training
- **Byte-Pair Encoding (BPE):** Uses BPE tokenization instead of WordPiece

**Performance (CoNLL-2003):**

- RoBERTa-base: ~92.8% F1 (vs BERT-base ~92.4%)
- RoBERTa-large: ~93.2% F1 (vs BERT-large ~92.8%)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RoBERTaNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a RoBERTa-NER model in ONNX inference mode. |
| `RoBERTaNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a RoBERTa-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

