---
title: "MisGANGenerator<T>"
description: "MisGAN generator for learning from incomplete data using dual generator/discriminator pairs for both data values and missingness patterns."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

MisGAN generator for learning from incomplete data using dual generator/discriminator
pairs for both data values and missingness patterns.

## For Beginners

MisGAN handles datasets with missing values:

Real datasets often have missing values (e.g., patients who skip blood tests,
customers who don't fill in all survey questions). MisGAN learns two things:

1. **What the complete data looks like** (the data generator)
2. **Which values tend to be missing** (the mask generator)

The data discriminator only sees observed values (masking out missing ones), just like
in real life. This forces the data generator to produce realistic complete rows, even
though it only gets feedback on partial observations.

If you provide custom layers in the architecture, those will be used directly
for the data generator network. If not, the network creates industry-standard
MisGAN layers based on the original research paper specifications.

Architecture:

Training loop (per batch):

Example usage:

## How It Works

MisGAN uses a dual-GAN architecture with four networks:

- **Data generator (G_x)**: Generates complete data rows from noise
- **Data discriminator (D_x)**: Judges masked data (only observed values) as real or fake
- **Mask generator (G_m)**: Generates realistic missingness patterns from noise
- **Mask discriminator (D_m)**: Judges missingness patterns as real or fake

Key innovations:

- **Masked discrimination**: D_x only sees data * mask (observed values), not complete rows
- **Missingness modeling**: G_m learns the missing data mechanism (MCAR/MAR/MNAR)
- **WGAN-GP training**: Both discriminators use Wasserstein distance with gradient penalty
- **Residual generators**: Skip connections for better gradient flow

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full autodiff and JIT compilation support

Reference: "MisGAN: Learning from Incomplete Data with GANs" (Li et al., ICLR 2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MisGANGenerator` | Initializes a new MisGAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the MisGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDataDiscriminator` | Builds the data discriminator network (auxiliary, not user-overridable). |
| `BuildMaskDiscriminator` | Builds the mask discriminator network (auxiliary, not user-overridable). |
| `BuildMaskGenerator` | Builds the mask generator network (auxiliary, not user-overridable). |
| `ClipWeights(IReadOnlyList<ILayer<>>)` | WGAN weight clipping: clamp each critic parameter to [-c, c] for Lipschitz continuity. |
| `CreateNewInstance` |  |
| `CreateRandomMask(Int32,Double)` | Creates a random binary mask vector where each element is 1 (observed) with probability (1 - missingRate). |
| `CreateStandardNormalVector(Int32)` | Creates a standard normal random vector of the given size. |
| `DataDiscriminatorForward(Tensor<>,Boolean)` | Forward pass through the data discriminator. |
| `DataGeneratorForward(Tensor<>)` | Forward pass through the data generator with residual connections. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the MisGAN network based on the provided architecture. |
| `MaskDiscriminatorForward(Tensor<>,Boolean)` | Forward pass through the mask discriminator. |
| `MaskGeneratorForward(Tensor<>)` | Forward pass through the mask generator with residual connections. |
| `PredictCore(Tensor<>)` |  |
| `SafeGradient(Tensor<>,Double)` | Sanitizes a gradient tensor by clamping NaN/Inf and clipping to max norm. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` |  |
| `TapeStepOver(GradientTape<>,Tensor<>,IReadOnlyList<ILayer<>>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>)` | Runs one optimizer step over the given sub-network's parameters from a tape-tracked loss. |
| `Train(Tensor<>,Tensor<>)` |  |
| `TryGetArchitectureInputShape` | MisGAN's `LayerBase` collection is the data GENERATOR — it consumes a latent z of size `EmbeddingDimension`, not the data row of size `Architecture.InputWidth`. |
| `UpdateParameters(Vector<>)` |  |

