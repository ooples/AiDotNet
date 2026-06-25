---
title: "GOGGLEGenerator<T>"
description: "GOGGLE generator that learns feature dependency structure via a graph neural network combined with a VAE framework for high-quality synthetic tabular data generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

GOGGLE generator that learns feature dependency structure via a graph neural network
combined with a VAE framework for high-quality synthetic tabular data generation.

## For Beginners

GOGGLE figures out which features relate to each other:

1. Learns a "graph" where connected features influence each other
2. Uses this graph to share information between related features
3. Generates new data where these relationships are preserved

If you provide custom layers in the architecture, those will be used for the
decoder (MLP) network. If not, the network creates standard decoder layers
based on the original paper specifications.

Example usage:

## How It Works

GOGGLE operates in three stages:

The adjacency matrix A is learned end-to-end alongside the encoder/decoder.
Regularization encourages A to be sparse and approximately acyclic (DAG-like).

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full forward/backward/update lifecycle

Reference: "GOGGLE: Generative Modelling for Tabular Data by Learning Relational Structure"
(Liu et al., ICLR 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GOGGLEGenerator` | Initializes a new GOGGLE generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `GoggleOptions` | Gets the GOGGLE-specific options. |
| `IsFitted` |  |
| `ParameterCount` | Total trainable parameter count = the layer parameters PLUS the learned adjacency tensor (a registered extra trainable tensor). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGoggleLossTape(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Negative ELBO + GOGGLE structure regularisers (paper Eq. |
| `CreateNewInstance` |  |
| `DecoderMlpLayers` | Returns the user-overridable decoder MLP sub-list of `Layers` (the slice between the encoder/projection heads and the `_decoderOutput`). |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `EncoderForwardTape(Tensor<>)` | Tape-connected GNN encoder + mean / logVar projection heads. |
| `EnsureSizedForInput(Tensor<>)` | Resizes encoder/decoder/adjacency to the actual input width when the caller has not yet invoked Fit. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetExtraTrainableTensors` | Expose the learned adjacency matrix so the tape-based trainer (BackwardAndStepOnPrecomputedLoss) collects it alongside the encoder/decoder layer parameters. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `GetParameters` |  |
| `InitializeLayers` | Initializes the decoder layers of the GOGGLE network. |
| `PredictCore(Tensor<>)` |  |
| `ProjectAdjacencyConstraints` | Re-projects `_adjacency` onto the valid GOGGLE soft-adjacency set after each optimizer step: off-diagonal entries are clamped to [0, 1] and the diagonal is forced to zero. |
| `RebuildAuxiliaryLayers` | Rebuilds auxiliary layers with actual data dimensions discovered during Fit(). |
| `ReparameterizeTape(Tensor<>,Tensor<>)` | Reparameterize z = μ + exp(0.5·logσ²) ⊙ ε with ε ~ N(0, I) sampled as a constant tensor (no gradient through ε). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `TraceTape(Tensor<>)` | tr(M) for a square 2-D tensor — tape-connected by summing the diagonal via a one-hot identity mask. |
| `Train(Tensor<>,Tensor<>)` | One paper-faithful ELBO + structure-regularisation training step: encoder → reparameterize → decoder → loss = ‖x-x̂‖² + KL + γ‖A‖₁ + ρ·h(A) then backprop through the encoder weights, mean/logVar heads, decoder weights, and the adjacency ten… |
| `TryGetArchitectureInputShape` | GOGGLE's `LayerBase` chain is the VAE decoder — Layer[0] takes the latent z (size LatentDimension), NOT the raw data row (size Architecture.InputWidth). |
| `UpdateParameters(Vector<>)` |  |

