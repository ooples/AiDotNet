---
title: "TabTransformerGenGenerator<T>"
description: "TabTransformer-Gen generator that uses column-wise contextual embeddings and masked prediction to generate realistic tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TabTransformer-Gen generator that uses column-wise contextual embeddings and masked
prediction to generate realistic tabular data.

## For Beginners

Think of this like a crossword puzzle:

- Each column is a clue that helps fill in other columns
- During training, we hide some columns and learn to fill them in
- During generation, we start with an empty puzzle and fill in one column at a time,

using already-filled columns as context for the remaining ones

If you provide custom layers in the architecture, those will be used for the
feed-forward network blocks. If not, the network creates standard TabTransformer
layers based on the original paper specifications.

Example usage:

## How It Works

TabTransformer-Gen treats each column as a "token" in a sequence:

**Training**: Mask random columns, predict their values from the unmasked context.
**Generation**: Start with all columns masked, iteratively unmask (predict) columns.

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full forward/backward/update lifecycle

Reference: "TabTransformer: Tabular Data Modeling Using Contextual Embeddings"
(Huang et al., 2020) — adapted for generation with masked prediction

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabTransformerGenGenerator` | Initializes a new TabTransformer-Gen generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `TransformerOptions` | Gets the TabTransformerGen-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyColumnMask(Vector<>,Int32)` | Returns a copy of `row` with `numMasked` randomly chosen columns zeroed out (the masked-prediction corruption). |
| `BuildLayers` | (Re)builds every trainable layer (column embeddings, per-layer Q/K/V projections, feed-forward blocks, LayerNorms, and column decoders) from the current `_numColumns` / `_colWidths` layout and registers them all in `Layers` so the tape-base… |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ExtractLayerReferences` | Re-binds the typed layer-reference lists (embeddings, per-block Q/K/V + FFN + norms, decoders) from the shared `Layers` collection, using the known build order and the current `_numColumns` / `_options` layout. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `ForwardForTraining(Tensor<>)` | The training forward — identical to `Predict` — overridden so the tape-based training path runs the column-token transformer rather than the default sequential walk over `Layers` (which would feed each layer's output to the next, ignoring t… |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `MultiHeadAttention(Tensor<>,Int32,Int32)` | Multi-head scaled-dot-product self-attention over the column-token sequence, computed entirely with tape-connected `Engine` ops. |
| `PredictCore(Tensor<>)` |  |
| `RunForward(Tensor<>)` | Tape-connected forward pass: embeds each column as a token, runs the column-token transformer (multi-head self-attention + feed-forward, each wrapped in a residual + LayerNorm), and decodes every column back to its value space. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `ShuffleInPlace(Int32[])` | Fisher-Yates shuffle using the generator's seeded RNG. |
| `SingleHeadAttention(Tensor<>,Tensor<>,Tensor<>,)` | softmax(Q·Kᵀ · scale)·V for a single head — all tape-connected. |
| `TryGetArchitectureInputShape` | Initializes the FFN layers of the TabTransformer-Gen network. |
| `UpdateParameters(Vector<>)` |  |

