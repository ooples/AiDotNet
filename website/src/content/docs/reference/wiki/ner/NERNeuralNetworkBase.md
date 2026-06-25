---
title: "NERNeuralNetworkBase<T>"
description: "Base class for NER-focused neural networks that can operate in both ONNX inference and native training modes."
section: "API Reference"
---

`Base Classes` · `AiDotNet.NER`

Base class for NER-focused neural networks that can operate in both ONNX inference and native training modes.

## For Beginners

NER neural networks read text and identify important entities like
people's names, company names, and places. This base class provides the shared foundation
that all NER models build upon.

You can use derived NER models in two ways:

1. **ONNX mode:** Load a pre-trained model for fast inference (identifying entities in text)
2. **Native mode:** Build and train a new model from scratch on your own labeled data

The model processes text as numerical vectors called "embeddings" - each word is represented
by a list of numbers that capture its meaning. Common embedding sources include GloVe
(100-300 dimensions), Word2Vec (300 dimensions), and BERT (768 dimensions).

## How It Works

This class extends `NeuralNetworkBase` to provide NER-specific functionality
while maintaining full integration with the AiDotNet neural network infrastructure. It serves
as the domain-level base class for all NER models, analogous to how
`VideoNeuralNetworkBase<T>` serves video models and
`AudioNeuralNetworkBase<T>` serves audio models.

NER (Named Entity Recognition) is a sequence labeling task where the model assigns a label
to each token in a text sequence. The labels identify whether each token is part of a named
entity (person, organization, location, etc.) or not. This is distinct from text classification
(which assigns a single label to an entire document) and from relation extraction (which
identifies relationships between entities).

This base class provides:

- Dual-mode support: ONNX inference for deployment and native training for model development
- Token embedding preprocessing utilities (normalization, batch handling)
- Sequence extraction helpers for batched input
- Common properties for embedding dimensions, label counts, and sequence lengths

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NERNeuralNetworkBase(NeuralNetworkArchitecture<>,ILossFunction<>,Double)` | Initializes a new instance of the NERNeuralNetworkBase class with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DefaultLossFunction` | Gets the default loss function for this model. |
| `EmbeddingDimension` | Gets or sets the embedding dimension for input token representations. |
| `IsOnnxMode` | Gets whether this model is running in ONNX inference mode. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length this model supports. |
| `NumLabels` | Gets or sets the number of entity label classes this model predicts. |
| `OnnxDecoder` | Gets or sets the ONNX decoder model for encoder-decoder NER architectures. |
| `OnnxEncoder` | Gets or sets the ONNX encoder model for encoder-decoder NER architectures. |
| `OnnxModel` | Gets or sets the single ONNX model for end-to-end NER architectures. |
| `SupportsTraining` | Gets whether this network supports training. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose(Boolean)` | Disposes of resources used by this model, including any loaded ONNX models. |
| `ExtractSequence(Tensor<>,Int32)` | Extracts a single sentence's embeddings from a batched tensor. |
| `Forward(Tensor<>)` | Performs a forward pass through the native neural network layers. |
| `NormalizeEmbeddings(Tensor<>)` | Normalizes token embeddings to unit length using L2 normalization. |
| `PostprocessOutput(Tensor<>)` | Postprocesses model output into label predictions. |
| `PreprocessTokens(Tensor<>)` | Preprocesses raw token embeddings for model input. |
| `RunOnnxInference(Tensor<>)` | Runs inference using the loaded ONNX model(s). |

