---
title: "IAutoregressiveMultimodalModel<T>"
description: "Defines the contract for autoregressive multimodal generation models that can generate tokens from any modality in an interleaved fashion."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for autoregressive multimodal generation models
that can generate tokens from any modality in an interleaved fashion.

## How It Works

This interface represents models like CM3Leon, Chameleon, or similar
that use a unified vocabulary across all modalities and generate
content token-by-token regardless of modality.

## Properties

| Property | Summary |
|:-----|:--------|
| `ModalityTokenCounts` | Gets the number of tokens reserved for each modality. |
| `VocabularySize` | Gets the vocabulary size (includes all modality tokens). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLoss(IEnumerable<MultimodalInput<>>,IEnumerable<MultimodalOutput<>>)` | Computes the loss for next-token prediction. |
| `Detokenize(IEnumerable<Int32>)` | Detokenizes token IDs back to multimodal outputs. |
| `GenerateNextToken(IEnumerable<MultimodalInput<>>,Double)` | Generates the next token given the context. |
| `GetNextTokenLogits(IEnumerable<MultimodalInput<>>)` | Gets token probabilities for next position. |
| `Tokenize(MultimodalInput<>)` | Tokenizes input into unified vocabulary tokens. |

