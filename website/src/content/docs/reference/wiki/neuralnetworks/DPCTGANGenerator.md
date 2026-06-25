---
title: "DPCTGANGenerator<T>"
description: "Differentially Private CTGAN (DP-CTGAN) for generating synthetic tabular data with formal (epsilon, delta)-differential privacy guarantees."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Differentially Private CTGAN (DP-CTGAN) for generating synthetic tabular data
with formal (epsilon, delta)-differential privacy guarantees.

## For Beginners

DP-CTGAN works exactly like CTGAN but with a "privacy shield":

**Standard CTGAN training step:****DP-CTGAN training step:**

If you provide custom layers in the architecture, those will be used directly
for the generator network. If not, the network creates industry-standard
CTGAN layers.

Example usage:

## How It Works

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

DP-CTGAN modifies the CTGAN training loop to provide differential privacy:

- **Per-sample gradient clipping**: Each sample's gradient is clipped to a fixed L2 norm
- **Gaussian noise**: Calibrated noise added to aggregated clipped gradients
- **Privacy accounting**: Tracks cumulative privacy cost using moments accountant
- **Early stopping**: Training halts when privacy budget is exhausted

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DPCTGANGenerator` | Initializes a new DP-CTGAN generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `CumulativeEpsilon` | Gets the cumulative privacy cost (epsilon) spent so far during training. |
| `IsFitted` |  |
| `Options` | Gets the DP-CTGAN-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClipAndNoiseGradient(ILayer<>)` | Clips a layer's gradient parameters to the specified L2 norm and adds Gaussian noise. |
| `ComputeGradientPenalty(Tensor<>,Tensor<>)` | WGAN-GP gradient penalty (Gulrajani et al. |
| `ComputeNoiseMultiplier(Int32,Int32)` | Computes the noise multiplier from the privacy budget if not manually specified. |
| `ComputePerExampleNoisedGradients(Tensor<>,Tensor<>,IReadOnlyList<Tensor<>>)` | DP-SGD gradient post-processing (Abadi et al. |
| `ComputeStepPrivacyCost(Int32,Int32)` | Computes the privacy cost (epsilon) for a single training step. |
| `ConcatInto(Vector<>,Vector<>,Vector<>)` | Concatenates two vectors into a pre-allocated destination buffer. |
| `CreateNewInstance` |  |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `FillFromTensor(Vector<>,Tensor<>)` | Copies tensor values into a pre-allocated vector buffer. |
| `FillRow(Vector<>,Matrix<>,Int32)` | Copies a row from a matrix into a pre-allocated vector. |
| `FillStandardNormal(Vector<>)` | Fills a pre-allocated vector with standard normal samples (Box-Muller). |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the DP-CTGAN generator to the provided real tabular data with differential privacy. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` |  |
| `InitializeTrainingBuffers(Int32)` | Pre-allocates all training buffers once to eliminate per-row GC pressure. |
| `PredictCore(Tensor<>)` |  |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainDiscriminatorStepBatchedDP(Matrix<>,Int32)` | Paper-faithful DP-SGD critic step (Abadi et al. |
| `TrainGeneratorStepBatched(Matrix<>,Int32)` | Non-DP generator step (Abadi et al. |
| `UpdateDiscriminatorParametersDP()` | Updates discriminator parameters with DP noise injection. |
| `UpdateParameters(Vector<>)` |  |

