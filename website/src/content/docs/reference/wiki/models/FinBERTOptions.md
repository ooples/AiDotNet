---
title: "FinBERTOptions<T>"
description: "Configuration options for FinBERT (Financial BERT) model."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for FinBERT (Financial BERT) model.

## For Beginners

FinBERT is designed specifically for financial text:

**The Key Insight:**
General-purpose sentiment models often misinterpret financial language. For example,
"shares fell 3%" is negative for the stock, but a general sentiment model might not
understand this. FinBERT is trained on financial text to understand such nuances.

**What Problems Does FinBERT Solve?**

- Sentiment analysis of financial news articles
- Analyzing SEC filings (10-K, 10-Q, 8-K)
- Processing earnings call transcripts
- Social media sentiment for stock prediction
- Document classification in financial contexts

**How FinBERT Works:**

1. **Pre-training:** BERT architecture trained on large text corpus
2. **Fine-tuning:** Further trained on financial-specific text and labels
3. **Tokenization:** Text is split into WordPiece tokens (subwords)
4. **Embedding:** Tokens are converted to dense vectors
5. **Transformer:** Self-attention captures context across the sequence
6. **Classification:** [CLS] token is used for sentiment prediction

**FinBERT Architecture:**

- Input: [CLS] token1 token2 ... tokenN [SEP]
- Embeddings: Token + Position + Segment embeddings
- Transformer: 12 layers of multi-head self-attention
- Output: Softmax over sentiment classes

**Key Benefits:**

- Understands financial language and terminology
- Captures context and nuance in financial text
- Pre-trained on large financial corpora
- State-of-the-art for financial sentiment analysis

## How It Works

FinBERT is a BERT model fine-tuned on financial text for sentiment analysis
and understanding of financial language.

**Reference:** Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", 2019.
https://arxiv.org/abs/1908.10063

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinBERTOptions` | Initializes a new instance of the `FinBERTOptions` class with default values. |
| `FinBERTOptions(FinBERTOptions<>)` | Initializes a new instance by copying from another instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionDropoutRate` | Gets or sets the attention-specific dropout rate. |
| `DropoutRate` | Gets or sets the dropout rate for regularization. |
| `FreezeBaseModel` | Gets or sets whether to freeze the base model during fine-tuning. |
| `HiddenDimension` | Gets or sets the hidden dimension of transformer layers. |
| `HiddenDropoutRate` | Gets or sets the hidden layer dropout rate. |
| `IntermediateDimension` | Gets or sets the intermediate (feed-forward) dimension. |
| `MaxPositionEmbeddings` | Gets or sets the maximum position embeddings. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length in tokens. |
| `NumAttentionHeads` | Gets or sets the number of attention heads. |
| `NumFineTuneLayers` | Gets or sets the number of layers to fine-tune (from the top). |
| `NumLayers` | Gets or sets the number of transformer layers. |
| `NumSentimentClasses` | Gets or sets the number of sentiment classes. |
| `PretrainedModelPath` | Gets or sets the path to pretrained model weights. |
| `TypeVocabSize` | Gets or sets the type vocabulary size (for sentence pairs). |
| `UsePretrainedWeights` | Gets or sets whether to use pretrained weights. |
| `VocabularySize` | Gets or sets the vocabulary size. |

