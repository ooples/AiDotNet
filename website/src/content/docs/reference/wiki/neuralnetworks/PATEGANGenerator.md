---
title: "PATEGANGenerator<T>"
description: "PATE-GAN generator for differentially private synthetic tabular data generation using the Private Aggregation of Teacher Ensembles (PATE) framework."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

PATE-GAN generator for differentially private synthetic tabular data generation using
the Private Aggregation of Teacher Ensembles (PATE) framework.

## For Beginners

PATE-GAN is like a game of telephone with added privacy:

1. Split real data among teachers (each sees only a small part)
2. Show generated data to all teachers: "Is this real or fake?"
3. Count votes and add random noise to the count
4. Tell the student the noisy answer
5. Generator tries to fool the student

The noise ensures no individual's information leaks through.

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates industry-standard
PATE-GAN layers based on the original research paper specifications.

Example usage:

## How It Works

PATE-GAN uses a teacher-student architecture for differential privacy:

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Support for custom layers via NeuralNetworkArchitecture
- Full forward → backward → update → reset lifecycle

Reference: "PATE-GAN: Generating Synthetic Data with Differential Privacy Guarantees"
(Jordon et al., ICLR 2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PATEGANGenerator` | Initializes a new PATE-GAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the PATE-GAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOutputActivations(Tensor<>)` | Applies per-column output activations: tanh for continuous values, softmax for mode indicators and categorical columns. |
| `BceLoss(Tensor<>,Double)` | Tape-connected binary cross-entropy for a single logit against a soft target in [0,1]. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GeneratorForward(Vector<>)` | Generator forward pass with residual connections, BatchNorm, and manual ReLU. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the generator layers from architecture or defaults. |
| `PredictCore(Tensor<>)` |  |
| `QueryTeachers(Vector<>)` | Queries the teacher ensemble with noisy aggregation for differential privacy. |
| `RebuildLayersWithActualDimensions` | Rebuilds default generator layers when actual data dimensions become known during Fit(). |
| `SampleLaplace(Double)` | Samples from the Laplace distribution using the inverse CDF method. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SetTrainingMode(Boolean)` | Propagates the training/inference mode to the auxiliary sub-networks (generator batch-norm and student dropout) that live outside the base `Layers` collection, so generation runs the generator batch-norm in inference mode with the learned r… |
| `StudentForward(Tensor<>,Boolean)` | Student discriminator forward pass with manual LeakyReLU and Dropout. |
| `TeacherForward(Int32,Tensor<>)` | Teacher discriminator forward pass with manual LeakyReLU. |
| `Train(Tensor<>,Tensor<>)` |  |
| `UpdateParameters(Vector<>)` |  |

