---
title: "BioBERTNER<T>"
description: "BioBERT-NER: Biomedical domain-specific BERT for Named Entity Recognition in biomedical text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

BioBERT-NER: Biomedical domain-specific BERT for Named Entity Recognition in biomedical text.

## For Beginners

BioBERT is BERT that has been additionally trained on millions of
biomedical research papers. It understands medical/scientific terminology better than general
BERT. Use BioBERT-NER when you need to extract entities from biomedical text (diseases,
drugs, genes, proteins, chemicals). Also consider PubMedBERT for even better biomedical NER.

## How It Works

BioBERT-NER (Lee et al., Bioinformatics 2020 - "BioBERT: a pre-trained biomedical language
representation model for biomedical text mining") is BERT pre-trained on large-scale biomedical
corpora for domain-specific NER tasks like gene, protein, disease, drug, and species recognition.

**Pre-training Data:**

- PubMed abstracts: ~4.5B words from 30M+ biomedical abstracts
- PMC full-text articles: ~13.5B words from open-access articles
- Initialized from BERT-base weights, then continued pre-training on biomedical text

**Biomedical NER Tasks:**

- **Gene/Protein NER:** BC2GM (87.5% F1), JNLPBA (79.3% F1)
- **Disease NER:** NCBI Disease (89.4% F1), BC5CDR-Disease (87.2% F1)
- **Drug/Chemical NER:** BC5CDR-Chemical (93.4% F1), BC4CHEMD (92.4% F1)
- **Species NER:** Species-800 (74.4% F1), LINNAEUS (88.2% F1)

**Why domain-specific pre-training matters:**
General BERT understands "IL-6" as just tokens, but BioBERT understands it as interleukin-6,
a cytokine. This domain knowledge comes from reading millions of biomedical papers during
pre-training, enabling BioBERT to recognize biomedical entities that general BERT misses.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BioBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a BioBERT-NER model in ONNX inference mode. |
| `BioBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a BioBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

