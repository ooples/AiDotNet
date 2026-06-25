---
title: "OCTGANGenerator<T>"
description: "OCT-GAN (One-Class Tabular GAN) generator for synthesizing minority-class tabular data using a one-class discriminator with Deep SVDD (Support Vector Data Description) objective."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

OCT-GAN (One-Class Tabular GAN) generator for synthesizing minority-class tabular data
using a one-class discriminator with Deep SVDD (Support Vector Data Description) objective.

## For Beginners

OCT-GAN is designed for imbalanced datasets where one class is rare
(e.g., fraud detection with 99% normal transactions and 1% fraud):

1. The **Generator** takes random noise and creates synthetic minority-class rows
2. The **Discriminator** learns a compact "sphere" around real minority samples
3. Real minority data maps close to the sphere's center (low SVDD score)
4. The generator learns to produce data that also maps near the center
5. A gradient penalty keeps training stable

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates standard layers.

Example usage:

## How It Works

OCT-GAN addresses the class imbalance problem by combining:

- **One-class discriminator**: Learns the hypersphere boundary of minority class data
- **Deep SVDD objective**: Maps real data close to a learned center, pushes fakes away
- **WGAN-GP training**: Wasserstein distance with gradient penalty for stable training
- **Residual generator**: Skip connections for better gradient flow
- **VGM normalization**: Handles multi-modal continuous distributions

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Support for custom layers via NeuralNetworkArchitecture
- Full forward, backward, update, reset lifecycle

Reference: "OCT-GAN: One-Class Tabular GAN for Imbalanced Data" (2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OCTGANGenerator(NeuralNetworkArchitecture<>,OCTGANOptions<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ILossFunction<>,Double)` | Initializes a new OCT-GAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the OCT-GAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` | Propagates the training/inference mode to the auxiliary sub-networks (generator batch-norm and discriminator dropout) that live outside the base `Layers` collection, so generation runs the generator batch-norm in inference mode with the lea… |
| `SvddDistSq(Tensor<>)` | Tape-connected squared distance of an embedding to the (constant) SVDD center. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `DefaultDataDimension` | Initializes a new OCT-GAN generator with default configuration. |

