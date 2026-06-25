---
title: "TextConditioningBase<T>"
description: "Base class for text conditioning modules used in diffusion models."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Diffusion.Conditioning`

Base class for text conditioning modules used in diffusion models.

## How It Works

Inherits `NeuralNetworkBase` so concrete subclasses use the
codebase's golden "Architecture.Layers ? : LayerHelper.CreateDefaultXxxTextLayers"
pattern — the same shape that the recent CLAP / AST / PANNs rewrites established.
The forward path walks the inherited `Layers`
list via `Tensor{`, so engine fast-paths,
gradient tape, and weight-streaming auto-detect (#1222, merged) all apply
automatically.

Tokenisation is delegated to an injected `ITokenizer` from
`src/Tokenization`. Each concrete subclass constructs its paper-
canonical tokenizer (CLIP byte-level BPE, T5 SentencePiece-Unigram,
Gemma SentencePiece, etc.) in its constructor; this base class never
fabricates token IDs from characters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextConditioningBase(NeuralNetworkArchitecture<>,ITokenizer,Int32,Int32,ILossFunction<>)` | Initialises a new text conditioning module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ConditioningType` |  |
| `EmbeddingDimension` |  |
| `MaxSequenceLength` |  |
| `ProducesPooledOutput` |  |
| `Tokenizer` | The injected tokenizer for this conditioner's text input. |
| `VocabSize` | Vocabulary size of the configured tokenizer. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildEncodingOptions` | Per-conditioner encoding options. |
| `BuildTokenTensor(List<TokenizationResult>)` | Packs `TokenizationResult`s into a `[batch, seqLen]` token-ID tensor for `EmbeddingLayer`. |
| `CreateDefaultLayers` | Layer-stack hook for concrete subclasses. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Encode(Tensor<>)` |  |
| `EncodeCompiled(Tensor<>,Func<Tensor<>>)` | Routes `eagerEncode` through the conditioner's compile host so the second + Nth call at the same input shape replays a cached compiled plan. |
| `EncodeCompiledAsync(Tensor<>,Func<Tensor<>>,CancellationToken)` | Async overload of `Tensor{`. |
| `EncodeText(Tensor<>,Tensor<>)` |  |
| `GetModelMetadata` |  |
| `GetPooledEmbedding(Tensor<>)` | Default pooling: mean over the sequence axis. |
| `GetUnconditionalEmbedding(Int32)` |  |
| `InitializeLayers` | Wires the layer stack at construction time. |
| `InvalidateConditionerCompiledPlans` | Invalidates any cached compiled plan after layer-graph mutations. |
| `MeanPool(Tensor<>)` | Mean-pools `[B, S, D]` to `[B, D]` via the engine's reduce kernel. |
| `RunLayerStack(Tensor<>)` | PyTorch-style direct forward pass: walk the inherited `Layers` list in order and let each layer's `Forward` handle its own shape contract. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Tokenize(String)` |  |
| `TokenizeBatch(String[])` |  |
| `UpdateParameters(Vector<>)` |  |

