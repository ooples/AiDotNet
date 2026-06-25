---
title: "TabLLMGenGenerator<T>"
description: "TabLLM-Gen generator that uses LLM-style schema-aware tokenization and autoregressive transformers to generate realistic tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TabLLM-Gen generator that uses LLM-style schema-aware tokenization and autoregressive
transformers to generate realistic tabular data.

## For Beginners

TabLLM-Gen works like an AI that fills in a form:

- It reads the form labels (column names and types)
- Fills in each field one by one, using previous answers to inform the next
- For example: after filling in "Age: 25", it knows to generate a realistic

income for a 25-year-old

If you provide custom layers in the architecture, those will be used directly
for the FFN blocks. Otherwise, the network creates standard layers based on
the original research paper specifications.

Example usage:

## How It Works

TabLLM-Gen processes a row as a sequence of (schema_token, value_token) pairs:

The model learns to generate value tokens conditioned on:

1. The column's schema tokens (name, type)
2. All previously generated column values

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full autodiff and JIT compilation support

Reference: "LLM-based Tabular Data Generation" (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabLLMGenGenerator` | Initializes a new TabLLM-Gen generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the TabLLM-Gen-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `CreateStandardNormalVector(Int32)` | Creates a vector of standard normal random values using Box-Muller transform. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetParameterVector` | Gets the current parameter vector from all layers. |
| `InitializeLayers` | Initializes the layers of the TabLLM-Gen network based on the provided architecture. |
| `PredictCore(Tensor<>)` |  |
| `RebuildAuxiliaryLayers` | Rebuilds the auxiliary transformer layers with actual vocabulary dimensions discovered during Fit(). |
| `SanitizeAndClipGradient(Tensor<>,Double)` | Sanitizes a gradient tensor by replacing NaN/Inf values with zero and applying gradient clipping. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

