---
title: "BERTNER<T>"
description: "BERT-NER: BERT (Bidirectional Encoder Representations from Transformers) with token classification head for Named Entity Recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

BERT-NER: BERT (Bidirectional Encoder Representations from Transformers) with token classification
head for Named Entity Recognition.

## For Beginners

BERT-NER is the most widely-used transformer NER model. It reads text
using "self-attention" where each word looks at every other word in the sentence to understand
context. This is much more powerful than BiLSTM-CRF because:

1. BERT understands context from both directions simultaneously at every layer
2. BERT was pre-trained on massive amounts of text, so it already "knows" language
3. Fine-tuning for NER only requires a small amount of labeled data

Think of BERT as a student who has read millions of documents and already understands
language well. Teaching it NER is like teaching that student a specific skill - it learns
quickly because it already has strong foundational knowledge.

Use BERT-NER when:

- You want state-of-the-art accuracy on standard NER benchmarks
- You have a GPU available for training/inference
- You can accept higher latency compared to BiLSTM-CRF

## How It Works

BERT-NER (Devlin et al., NAACL 2019 - "BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding") fine-tunes a pre-trained BERT model for NER by adding a token
classification head. BERT introduced the masked language model (MLM) and next sentence prediction
(NSP) pre-training objectives, enabling deep bidirectional context understanding.

**Key Features:**

- **Bidirectional self-attention:** Unlike GPT (left-to-right only), BERT attends to all

positions simultaneously, capturing both left and right context at every layer

- **WordPiece tokenization:** Handles out-of-vocabulary words by splitting them into subword

units (e.g., "unforgettable" -> "un", "##forget", "##table")

- **Pre-trained representations:** Learns general language understanding from massive corpora

(BookCorpus + English Wikipedia), then fine-tuned for NER

- **[CLS] and [SEP] tokens:** Special tokens mark sequence boundaries

**Architecture (BERT-base):**

- 12 transformer layers, 768 hidden units, 12 attention heads (~110M parameters)
- Input: WordPiece token embeddings + positional embeddings + segment embeddings
- Output per token: 768-dimensional contextual representation
- NER head: Linear projection (768 -> num_labels) with optional CRF

**Performance (CoNLL-2003 English NER):**

- BERT-base: ~92.4% F1 (compared to 91.2% for BiLSTM-CRF)
- BERT-large: ~92.8% F1

**SubWord Alignment for NER:**
When a word is split into multiple WordPiece tokens, only the first subword token's prediction
is used as the entity label for the whole word. For example:
"Washington" -> ["Wash", "##ington"] -> only "Wash" gets a label, "##ington" is ignored.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a BERT-NER model in ONNX inference mode. |
| `BERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

