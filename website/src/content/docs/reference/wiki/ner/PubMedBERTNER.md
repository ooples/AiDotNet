---
title: "PubMedBERTNER<T>"
description: "PubMedBERT-NER: PubMed domain-specific BERT pre-trained from scratch on biomedical text for NER."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

PubMedBERT-NER: PubMed domain-specific BERT pre-trained from scratch on biomedical text for NER.

## For Beginners

PubMedBERT is like BioBERT but even more specialized. While BioBERT
started as a general English model and then read biomedical papers, PubMedBERT was built
entirely from biomedical text from day one. It has a custom vocabulary designed for medical
terms, making it the best choice for biomedical NER tasks like extracting drug names,
disease names, gene names, and protein names from research papers and clinical text.

## How It Works

PubMedBERT-NER (Gu et al., ACL 2021 - "Domain-Specific Language Model Pretraining for
Biomedical Natural Language Processing") is BERT pre-trained entirely from scratch on
PubMed abstracts, using a domain-specific vocabulary. Unlike BioBERT (which initializes
from general BERT), PubMedBERT builds its vocabulary and model weights exclusively from
biomedical text.

**Key Difference from BioBERT:**

- **BioBERT:** Starts from general BERT weights + vocabulary, then continues pre-training

on biomedical text. The vocabulary still contains many general-domain tokens.

- **PubMedBERT:** Pre-trained from scratch on PubMed only. Its vocabulary is 100%

biomedical, so terms like "immunoglobulin" or "phosphorylation" are single tokens
instead of being split into many subwords.

**Pre-training Data:**

- PubMed abstracts only: ~3.1B words from 14M+ abstracts
- Custom WordPiece vocabulary built from PubMed text (28,895 tokens)
- No general-domain text (no Wikipedia, no BookCorpus)

**Why From-Scratch Pre-training Matters:**
The vocabulary is the key. BioBERT inherits BERT's general vocabulary where "phosphorylation"
becomes ["ph", "##os", "##pho", "##ry", "##lation"] (5 tokens). In PubMedBERT, it's a
single token "phosphorylation". This means:

- More efficient encoding of biomedical text
- Each token carries more semantic meaning
- Better representation of domain-specific terms

**Performance (Biomedical NER):**

- BC5CDR Chemical: ~94.1% F1 (vs BioBERT ~93.4%)
- BC5CDR Disease: ~87.8% F1 (vs BioBERT ~87.2%)
- NCBI Disease: ~89.8% F1 (vs BioBERT ~89.4%)
- BC2GM (Gene/Protein): ~88.1% F1 (vs BioBERT ~87.5%)
- JNLPBA: ~80.1% F1 (vs BioBERT ~79.3%)

PubMedBERT consistently outperforms BioBERT across all biomedical NER benchmarks,
demonstrating that from-scratch domain-specific pre-training is superior to continued
pre-training from a general model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PubMedBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a PubMedBERT-NER model in ONNX inference mode. |
| `PubMedBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a PubMedBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

