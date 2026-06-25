---
title: "FinBERTNER<T>"
description: "FinBERT-NER: Financial domain BERT for Named Entity Recognition in financial text."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NER.TransformerBased`

FinBERT-NER: Financial domain BERT for Named Entity Recognition in financial text.

## For Beginners

FinBERT is BERT trained on financial documents like SEC filings,
earnings reports, and financial news. It understands financial jargon, company names,
and financial metrics better than general BERT. Use FinBERT-NER for extracting entities
from financial text: company names, ticker symbols, financial figures, regulatory terms.

## How It Works

FinBERT-NER (Araci, 2019 - "FinBERT: Financial Sentiment Analysis with Pre-trained Language
Models"; Yang et al., IJCAI 2020 - "FinBERT: A Pretrained Language Model for Financial
Communications") is BERT further pre-trained on large-scale financial corpora for
domain-specific NLP tasks including financial NER.

**Pre-training Data:**

- Financial news articles (Reuters, Bloomberg)
- SEC filings (10-K, 10-Q, 8-K reports)
- Financial analyst reports
- Earnings call transcripts
- ~4.9B tokens of financial text

**Financial NER Entity Types:**

- **Company:** Apple Inc., Goldman Sachs, Tesla
- **Person:** CEOs, CFOs, board members, analysts
- **Financial Metric:** Revenue, EBITDA, P/E ratio, market cap
- **Currency/Amount:** $1.5 billion, EUR 200 million
- **Date/Period:** Q3 2023, fiscal year 2024, YoY
- **Regulation:** Dodd-Frank, Basel III, SOX
- **Index/Ticker:** S&P 500, NASDAQ, AAPL

**Why Financial NER Needs Domain Models:**
Financial text has unique challenges: "Apple" is a company (not fruit), "bearish" describes
market sentiment (not an animal), and "spread" refers to yield difference (not butter).
FinBERT's financial pre-training resolves these ambiguities that confuse general BERT.

**Performance:**

- Financial NER: ~90-92% F1 (vs general BERT ~85-87% on financial text)
- Financial sentiment analysis: ~88% accuracy (state-of-the-art)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinBERTNER(NeuralNetworkArchitecture<>,String,TransformerNEROptions)` | Creates a FinBERT-NER model in ONNX inference mode. |
| `FinBERTNER(NeuralNetworkArchitecture<>,TransformerNEROptions,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Creates a FinBERT-NER model in native training mode. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |

