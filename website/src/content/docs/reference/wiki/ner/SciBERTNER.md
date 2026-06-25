---
title: "SciBERTNER<T>"
description: "SciBERT-NER: Scientific domain BERT for Named Entity Recognition in scientific literature."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

SciBERT-NER: Scientific domain BERT for Named Entity Recognition in scientific literature.

## For Beginners

SciBERT is BERT pre-trained entirely on scientific papers. It has a
specialized vocabulary designed for scientific terms. Use SciBERT-NER for extracting entities
from scientific literature, especially when working across both biomedical and computer
science domains. For purely biomedical NER, BioBERT or PubMedBERT may perform slightly better.

## How It Works

SciBERT-NER (Beltagy et al., EMNLP 2019 - "SciBERT: A Pretrained Language Model for
Scientific Text") is BERT pre-trained from scratch on 1.14M scientific papers from
Semantic Scholar, covering computer science (18%) and biomedical (82%) domains.

**Key Differences from BioBERT:**

- **Pre-trained from scratch:** SciBERT uses its own vocabulary (SciVocab) built from

scientific text, while BioBERT initializes from BERT's vocabulary

- **Domain-specific vocabulary:** SciVocab (31K tokens) contains scientific terms that

BERT's WordPiece vocabulary would split into many subwords

- **Broader scientific coverage:** Includes computer science papers, not just biomedical

**Scientific NER Tasks:**

- **SciERC:** Scientific entities (Task, Method, Metric, Material, Generic, OtherScientificTerm)
- **JNLPBA:** Biomedical entities (Protein, DNA, RNA, Cell Line, Cell Type) - 77.3% F1
- **BC5CDR:** Chemical and disease entities - 90.0% F1
- **NCBI Disease:** Disease mention recognition - 88.6% F1

**Architecture:**
Same as BERT-base (12 layers, 768 hidden, 12 heads, 110M params) but with SciVocab
tokenization. The key insight is that domain-specific vocabulary matters: "immunoglobulin"
is one token in SciVocab but ["im", "##mun", "##og", "##lo", "##bul", "##in"] in BERT.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SciBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a SciBERT-NER model in ONNX inference mode. |
| `SciBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SciBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

