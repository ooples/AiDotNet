---
title: "REaLTabFormerGenerator<T>"
description: "REaLTabFormer generator using GPT-2 style autoregressive transformer for synthetic tabular data generation by treating columns as a sequence of tokens."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

REaLTabFormer generator using GPT-2 style autoregressive transformer for synthetic
tabular data generation by treating columns as a sequence of tokens.

## For Beginners

REaLTabFormer generates data like writing a sentence word by word:

If you provide custom layers in the architecture, those will be used directly
for the FFN blocks. If not, the network creates the standard REaLTabFormer
architecture based on the original research paper specifications.

Example usage:

## How It Works

REaLTabFormer architecture:

- **Tokenization**: Continuous values are binned; categoricals are integer-encoded
- **Embedding**: Token embeddings + positional encodings for each column position
- **Transformer**: Causal self-attention (each column can only see previous columns)
- **Output**: Per-column classification heads predict the token for each column
- **Training**: Cross-entropy loss on next-column prediction (teacher forcing)
- **Generation**: Autoregressive sampling left-to-right with temperature

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full autodiff and JIT compilation support

Reference: "REaLTabFormer" (Solatorio and Dupriez, 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `REaLTabFormerGenerator` | Initializes a new REaLTabFormer generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the REaLTabFormer-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildOutputHeads` | Builds per-column output heads (auxiliary, depend on data columns and vocab sizes). |
| `BuildTransformerAttention` | Builds the transformer attention layers (auxiliary, not user-overridable). |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the REaLTabFormer network. |
| `PredictCore(Tensor<>)` |  |
| `SafeGradient(Tensor<>,Double)` | Sanitizes a gradient tensor by clamping NaN/Inf and clipping to max norm. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

