---
title: "SECBertNER<T>"
description: "SEC-BERT-NER: Securities and Exchange Commission domain BERT for NER in regulatory filings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

SEC-BERT-NER: Securities and Exchange Commission domain BERT for NER in regulatory filings.

## For Beginners

SEC-BERT is a specialized version of BERT trained on SEC regulatory
filings (10-K, 10-Q reports that public companies must file). It excels at extracting
entities from these highly structured financial documents. Use SEC-BERT-NER when processing
SEC filings, XBRL documents, or regulatory financial text.

## How It Works

SEC-BERT-NER (Loukas et al., EMNLP 2022 - "FiNER: Financial Numeric Entity Recognition for
XBRL Tagging") and related work uses BERT pre-trained specifically on SEC EDGAR filings
for financial regulatory NER.

**Pre-training Data:**

- SEC EDGAR filings: 10-K, 10-Q, 8-K, proxy statements, prospectuses
- ~250K filing documents spanning 1993-2023
- Financial regulatory language with strict formatting conventions

**SEC-Specific NER Entity Types:**

- **XBRL Tags:** us-gaap:Revenue, us-gaap:NetIncomeLoss, dei:EntityRegistrantName
- **Filing Entity:** Registrant names, CIK numbers
- **Monetary Values:** Amounts in SEC-mandated formats ($X,XXX)
- **Date References:** Filing dates, fiscal periods, reporting dates
- **Regulatory References:** Item numbers, exhibit references, rule citations
- **Financial Statements:** Balance sheet items, income statement items

**Why SEC-specific NER matters:**
SEC filings follow strict formatting rules and use specialized terminology. A filing might
reference "Item 7" (MD&A section) or "Exhibit 31.1" (SOX certification). SEC-BERT
understands these domain-specific patterns that general financial models miss.

**Performance:**

- XBRL entity recognition: ~85-88% F1 (vs general BERT ~75-78%)
- Filing entity extraction: ~91-93% F1

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SECBertNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a SEC-BERT-NER model in ONNX inference mode. |
| `SECBertNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a SEC-BERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

