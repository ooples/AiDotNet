---
title: "TimeGANGenerator<T>"
description: "TimeGAN generator for synthesizing realistic time-series tabular data while preserving temporal dynamics using an embedding-supervisor-adversarial training framework."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TimeGAN generator for synthesizing realistic time-series tabular data while preserving
temporal dynamics using an embedding-supervisor-adversarial training framework.

## For Beginners

TimeGAN works by:

1. Learning to compress time-series into a simpler space (embedding)
2. Learning how data moves through time in that space (supervisor)
3. Learning to generate realistic fake data using both spatial and temporal info

Example usage:

## How It Works

TimeGAN uses five jointly trained components in a shared latent space:

Training has three phases:

1. **Embedding phase**: Train embedder + recovery to reconstruct data
2. **Supervised phase**: Train supervisor to predict next-step embeddings
3. **Joint phase**: Train all 5 components together with combined losses

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layers = generator network (user-overridable via Architecture)
- Auxiliary networks (embedder, recovery, supervisor, discriminator) are internal

Reference: "Time-series Generative Adversarial Networks" (Yoon et al., NeurIPS 2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeGANGenerator` | Initializes a new TimeGAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `TimeGanOptions` | Gets the TimeGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDiscLayerList` | Builds a combined list of discriminator layers (dense + dropout + output) for gradient-penalty and related analyses. |
| `BuildFlattenedSequenceBatch(List<Matrix<>>,Int32,Int32)` | Flattens timesteps across a slice of sequences into a single `[batchSize, dataWidth]` tensor for batched processing. |
| `BuildPairedSequenceBatch(List<Matrix<>>,Int32,Int32)` | Builds the paired (x_t, x_{t+1}) batches from sequence slices for the supervisor's next-step prediction objective. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the generator layers (Layers = generator network, user-overridable). |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainDiscriminatorStepBatched(List<Matrix<>>,Int32,Int32)` | Paper-faithful TimeGAN Phase 3 critic step (Yoon et al. |
| `TrainEmbeddingStepBatched(List<Matrix<>>,Int32,Int32)` | Paper-faithful TimeGAN Phase 1 (Yoon et al. |
| `TrainGeneratorStepBatched(List<Matrix<>>,Int32,Int32)` | Paper-faithful TimeGAN Phase 3 generator + supervisor joint step: non-saturating adversarial loss + supervised next-step loss in the embedded space (Yoon et al. |
| `TrainSupervisedStepBatched(List<Matrix<>>,Int32,Int32)` | Paper-faithful TimeGAN Phase 2 (Yoon et al. |
| `UpdateParameters(Vector<>)` |  |

