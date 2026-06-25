---
title: "MedSynthGenerator<T>"
description: "MedSynth generator for privacy-preserving medical tabular data synthesis using a VAE/GAN hybrid with clinical validity constraints and optional differential privacy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

MedSynth generator for privacy-preserving medical tabular data synthesis using a
VAE/GAN hybrid with clinical validity constraints and optional differential privacy.

## For Beginners

MedSynth ensures generated medical data is:

1. Realistic (VAE reconstruction + GAN adversarial training)
2. Valid (no impossible lab values or vital signs)
3. Private (optional differential privacy protection)

If you provide custom layers in the architecture, those will be used directly
for the decoder network. Otherwise, standard layers are created.

Example usage:

## How It Works

MedSynth combines VAE and GAN approaches with medical domain constraints:

Training alternates between three objectives:

1. **VAE loss**: Reconstruction + KL divergence + constraint violation penalty
2. **Discriminator loss**: BCE on real vs fake samples
3. **Adversarial loss**: Non-saturating generator loss through discriminator input gradients

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Full autodiff and JIT compilation support

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MedSynthGenerator` | Initializes a new MedSynth generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the MedSynth-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDiscLayerList` | Builds a combined list of discriminator layers (dense + dropout + output) for gradient-penalty and related analyses. |
| `BuildRealAndFakeBatches(Matrix<>,Int32,Int32)` | Builds [batchSize, dataWidth] real + fake batches for one DP-SGD critic step. |
| `CreateNewInstance` |  |
| `CreateStandardNormalVector(Int32)` | Creates a vector of standard normal random values using Box-Muller transform. |
| `DecoderForwardBatched(Tensor<>,Boolean)` | Batched, tape-tracked decoder forward (replaces per-row `Boolean)`). |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiscriminatorForwardBatched(Tensor<>,Boolean)` | Batched, tape-tracked discriminator forward. |
| `EncoderForwardBatched(Tensor<>)` | Batched, tape-tracked encoder forward. |
| `EnsureSizedForInput(Tensor<>)` | Before Fit() supplies the real transformed width, adapt the encoder/decoder/ discriminator layout to the actual input width so the model is a valid network for any 1-D input — the generated ModelFamily tests call Train()/Predict() directly… |
| `ExtractMedSynthLayerReferences` | Extracts private layer references as aliases from the unified Layers list. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` |  |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `ForwardForTraining(Tensor<>)` | Training forward — the same deterministic VAE reconstruction as `Predict`, overridden so the tape-based `Tensor{` path trains the encoder + latent-mean head + decoder rather than walking the full Layers list (which also holds the discrimina… |
| `Generate(Int32,Vector<>,Vector<>)` |  |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the MedSynth network based on the provided architecture. |
| `LogSigmoid(Tensor<>)` | Numerically-stable `log(sigmoid(x))` = `-softplus(-x)`, expressed via tape-tracked Engine ops so backprop flows correctly through the BCE-with-logits loss. |
| `PredictCore(Tensor<>)` |  |
| `RebuildLayersWithActualDimensions` | Rebuilds all layers with actual data dimensions discovered during Fit(). |
| `ReconstructForward(Tensor<>)` | Deterministic VAE reconstruction (encode → latent mean → decode), tape-connected and shared by `Predict` and `Tensor{`. |
| `SanitizeAndClipGradient(Tensor<>,Double)` | Sanitizes a gradient tensor by replacing NaN/Inf values with zero and applying gradient clipping. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `Train(Tensor<>,Tensor<>)` |  |
| `TrainDiscriminatorStepBatched(Matrix<>,Int32,Int32,Double)` | Paper-faithful DP-SGD discriminator step (Abadi et al. |
| `TrainDiscriminatorStepPerExampleDPSGD(Matrix<>,Int32,Int32,Double)` | Per-example DP-SGD discriminator step (Abadi et al. |
| `TrainGeneratorStepBatched(Int32)` | Paper-faithful generator/decoder step for MedSynth (no DP — adversarial gradients flow into the generator which never directly touches real data, covered by the data-processing inequality). |
| `TrainVaeStepBatched(Matrix<>,Int32,Int32)` | VAE-half of the MedSynth (VAE+GAN hybrid) training loop. |
| `UpdateParameters(Vector<>)` |  |

