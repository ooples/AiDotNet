---
title: "CTABGANPlusGenerator<T>"
description: "CTAB-GAN+ generator for high-quality synthetic tabular data with auxiliary classifier discriminator, mixed-type encoder, and information loss."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

CTAB-GAN+ generator for high-quality synthetic tabular data with auxiliary classifier
discriminator, mixed-type encoder, and information loss.

## For Beginners

CTAB-GAN+ works like CTGAN but with smarter training:

**Architecture:****Three-part loss for discriminator:**

1. Wasserstein loss (real vs fake)
2. Auxiliary classifier loss (correct class prediction)
3. Gradient penalty (training stability)

**Three-part loss for generator:**

1. Fool discriminator (maximize fake score)
2. Correct class via auxiliary classifier
3. Information loss (match data statistics)

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates industry-standard
CTAB-GAN+ layers based on the original research paper specifications.

Example usage:

## How It Works

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

CTAB-GAN+ extends the CTGAN architecture with:

- **ACGAN-style discriminator**: Outputs both real/fake score and class label predictions
- **Information loss**: Penalizes statistical divergence between real and generated data
- **Log-frequency encoding**: Better handling of long-tail categorical distributions
- **Conditional training**: Same training-by-sampling as CTGAN

Reference: "CTAB-GAN: Effective Table Data Synthesizing" (Zhao et al., 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CTABGANPlusGenerator` | Initializes a new CTAB-GAN+ generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the CTAB-GAN+-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddInformationLossGradient(Tensor<>,Vector<>,Matrix<>)` | Adds information loss gradient to the generator gradient. |
| `ApplyOutputActivations(Tensor<>)` | Applies tanh to continuous columns and softmax to categorical/mode columns. |
| `BuildDiscriminatorDimensionMap(Int32)` | Builds a dimension map for the discriminator layers to support manual backward pass. |
| `ComputeClassifierGradient(Tensor<>,Vector<>)` | Computes the gradient of the auxiliary classifier loss with respect to logits. |
| `ConcatInto(Vector<>,Vector<>,Vector<>)` | Concatenates two vectors into a pre-allocated destination buffer. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorForward(Tensor<>,Boolean)` | Runs the discriminator forward pass, returning critic score, classifier logits, and hidden output. |
| `FillFromTensor(Vector<>,Tensor<>)` | Copies tensor values into a pre-allocated vector buffer. |
| `FillRow(Vector<>,Matrix<>,Int32)` | Copies a row from a matrix into a pre-allocated vector. |
| `FillStandardNormal(Vector<>)` | Fills a pre-allocated vector with standard normal samples (Box-Muller). |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the CTAB-GAN+ generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows. |
| `GeneratorForwardWithResidual(Tensor<>)` | Generator forward pass with CTGAN-style residual connections: Each hidden layer receives [previous_output, original_input]. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `IdentifyTargetColumn(IReadOnlyList<ColumnMetadata>)` | Identifies the target column for the auxiliary classifier. |
| `InitializeLayers` | Initializes the layers of the CTAB-GAN+ network based on the provided architecture. |
| `InitializeTrainingBuffers(Int32)` | Pre-allocates all training buffers once to eliminate per-row GC pressure. |
| `PredictCore(Tensor<>)` | Runs the generator forward pass to create synthetic data from a noise input. |
| `RebuildLayersWithActualDimensions(Int32,Int32,Int32)` | Rebuilds the default generator and discriminator layers with actual data dimensions discovered during Fit(). |
| `SafeGradient(Tensor<>,Double)` | Applies NaN sanitization and gradient norm clipping in a single operation. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` | Trains the CTAB-GAN+ network using the provided input and expected output. |
| `UpdateDiscriminatorParameters()` | Updates discriminator parameters with a given learning rate. |
| `UpdateGeneratorParameters()` | Updates generator parameters with a given learning rate. |
| `UpdateParameters(Vector<>)` |  |

