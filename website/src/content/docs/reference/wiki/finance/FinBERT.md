---
title: "FinBERT<T>"
description: "FinBERT (Financial BERT) model for financial sentiment analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Finance.NLP`

FinBERT (Financial BERT) model for financial sentiment analysis.

## For Beginners

FinBERT solves a key problem in financial NLP:

**The Key Insight:**
General-purpose sentiment models often misinterpret financial language. For example,
"shares dropped 5% on earnings miss" is clearly negative for the stock, but a general
sentiment model might not understand this. FinBERT is trained specifically on financial
text to understand such nuances.

**What Problems Does FinBERT Solve?**

- Sentiment analysis of financial news articles
- Processing SEC filings (10-K, 10-Q, 8-K) for sentiment signals
- Analyzing earnings call transcripts for tone
- Social media sentiment monitoring for trading signals
- Document classification in financial contexts

**How FinBERT Works:**

1. **Tokenization:** Text is split into WordPiece tokens (subwords)
2. **Embedding:** Tokens are converted to dense vectors with position information
3. **Transformer:** 12 layers of bidirectional self-attention capture context
4. **Classification:** [CLS] token embedding is used for sentiment prediction
5. **Output:** Softmax over classes: Positive, Negative, Neutral

**FinBERT Architecture:**

- Input: [CLS] token1 token2 ... tokenN [SEP]
- Embeddings: Token + Position + Segment embeddings (768-dim)
- Transformer: 12 layers, 12 heads, 768 hidden dim
- Output: 3-class softmax (negative, neutral, positive)

**Key Benefits:**

- Understands financial language and terminology
- Captures context across the entire input sequence
- Pre-trained on large financial corpora
- State-of-the-art accuracy on financial sentiment benchmarks

## How It Works

FinBERT is a BERT model fine-tuned on financial text for sentiment analysis.
It understands financial terminology and context to accurately classify
sentiment in financial news, SEC filings, earnings calls, and other financial text.

**Reference:** Araci, "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models", 2019.
https://arxiv.org/abs/1908.10063

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FinBERT(NeuralNetworkArchitecture<>,FinBERTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the FinBERT model in native mode for training. |
| `FinBERT(NeuralNetworkArchitecture<>,String,FinBERTOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>)` | Initializes a new instance of the FinBERT model in ONNX mode for inference. |

## Properties

| Property | Summary |
|:-----|:--------|
| `HiddenDimension` |  |
| `MaxSequenceLength` |  |
| `NumAttentionHeads` | Gets the number of attention heads. |
| `NumLayers` | Gets the number of transformer layers. |
| `NumSentimentClasses` |  |
| `SupportsTraining` |  |
| `UseNativeMode` |  |
| `VocabularySize` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AnalyzeSentiment(String[])` | Analyzes sentiment from raw text strings. |
| `AnalyzeSentiment(Tensor<>)` | Analyzes sentiment from tokenized input. |
| `ApplySoftmax(Tensor<>)` | Applies softmax to convert logits to probabilities. |
| `CreateNewInstance` | Creates a new instance with the same configuration. |
| `DeserializeNetworkSpecificData(BinaryReader)` | Deserializes FinBERT-specific data. |
| `Detokenize(Int32[])` | Converts token IDs back to text. |
| `Dispose(Boolean)` | Disposes managed resources. |
| `ExtractLayerReferences` | Extracts references to specific layer types for direct access. |
| `Forward(Tensor<>)` | Performs the forward pass through all layers. |
| `ForwardForTraining(Tensor<>)` | Tape-aware training forward. |
| `ForwardNative(Tensor<>)` | Performs native mode forward pass. |
| `ForwardOnnx(Tensor<>)` | Performs ONNX mode forward pass. |
| `GetEmbeddings(Tensor<>)` | Gets embeddings (vector representations) for input tokens. |
| `GetFinancialMetrics` | Gets financial-specific NLP metrics from the model. |
| `GetModelMetadata` | Gets metadata about the FinBERT model. |
| `GetOptions` |  |
| `GetSequenceEmbedding(Tensor<>)` | Gets the [CLS] token embedding representing the entire input sequence. |
| `InitializeBasicVocabulary` | Initializes a basic vocabulary for demonstration purposes. |
| `InitializeLayers` | Initializes the neural network layers for FinBERT. |
| `PredictCore(Tensor<>)` | Performs forward prediction on the input tensor. |
| `SerializeNetworkSpecificData(BinaryWriter)` | Serializes FinBERT-specific data. |
| `TokenIdsToTensor(Int32[])` | Converts token ID array to tensor. |
| `Tokenize(String,Nullable<Int32>)` | Tokenizes raw text into token IDs. |
| `Train(Tensor<>,Tensor<>)` | Trains the FinBERT model on a batch of input-target pairs. |
| `UpdateParameters(Vector<>)` | Updates parameters using the provided gradients. |
| `ValidateCustomLayers(List<ILayer<>>)` | Validates custom layers provided by the user. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_vocabulary` | Simple vocabulary mapping for demonstration. |

