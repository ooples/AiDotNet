---
title: "TabSynGenerator<T>"
description: "TabSyn generator combining VAE pretraining with latent diffusion for state-of-the-art synthetic tabular data generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

TabSyn generator combining VAE pretraining with latent diffusion for state-of-the-art
synthetic tabular data generation.

## For Beginners

TabSyn is a two-step generator:

**Step 1 - VAE Training (learning to compress):****Step 2 - Diffusion Training (learning the latent distribution):****Generation:**

If you provide custom layers in the architecture, those will be used directly
for the VAE encoder. If not, the network creates industry-standard
TabSyn encoder layers based on the original research paper specifications.

Example usage:

## How It Works

TabSyn trains in two phases:

1. **VAE Phase**: Encoder-decoder learns latent representation of tabular data
2. **Diffusion Phase**: Gaussian diffusion model learns the distribution of latent codes

Generation: z ~ DiffusionModel -> decoded = VAEDecoder(z) -> inverse transform

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "TabSyn: Bridging the Gap" (Zhang et al., NeurIPS 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TabSynGenerator` | Initializes a new TabSyn generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the TabSyn-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOutputActivations(Tensor<>)` | Applies per-column output activations: tanh for continuous values, softmax for modes/categories. |
| `ApplySoftmax(Tensor<>,Tensor<>,Int32,Int32)` | Applies stable softmax activation to a contiguous block of tensor elements. |
| `BuildAuxiliaryNetworks(Int32)` | Builds the auxiliary sub-networks (mean/logvar heads, decoder, diffusion MLP, timestep projection) that live outside the base `Layers` collection. |
| `ComputeElboLossTape(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Tape-connected negative ELBO: per-column reconstruction (tanh+MSE for the continuous value, softmax+cross-entropy for mode indicators and categorical one-hots — the CTGAN/TVAE transform loss) plus the Gaussian KL to N(0,I). |
| `CreateNewInstance` |  |
| `CreateTimestepEmbedding(Int32)` | Creates a sinusoidal timestep embedding for the diffusion model. |
| `DecoderForward(Tensor<>)` | Runs the VAE decoder forward pass to reconstruct data from latent code. |
| `DeriveShapeWithLastDim(Int32[],Int32)` | Derives a tensor shape that preserves the rank of the reference but replaces the last dimension. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `DiffusionMLPForward(Vector<>,Vector<>)` | Runs the diffusion denoiser MLP forward pass. |
| `DiffusionMLPForwardOnTape(Vector<>,Vector<>)` | Tape-connected latent-diffusion denoiser forward returning the predicted noise tensor. |
| `EncodeAllData(Matrix<>)` | Encodes all training data rows to their mean latent representations. |
| `EncoderForwardDefault(Tensor<>)` | Runs the default encoder forward pass (sequential through all LayerHelper-created layers). |
| `EncoderForwardOnTape(Tensor<>)` | Tape-connected encoder forward (Layers chain) for the VAE training step. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the TabSyn generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates new synthetic tabular data rows. |
| `GetFeatureImportance` |  |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the TabSyn network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Runs the VAE encoder forward pass on input data. |
| `RebuildLayersWithActualDimensions(Int32)` | Rebuilds encoder, decoder, and diffusion layers with actual data dimensions discovered during Fit(). |
| `Reparameterize(Tensor<>,Tensor<>)` | VAE reparameterization trick: z = mean + std * epsilon, where epsilon ~ N(0,1). |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SplitEncoderOutput(Tensor<>)` | Extracts mean and log-variance from the encoder output. |
| `TapeStepOver(GradientTape<>,Tensor<>,IEnumerable<ILayer<>>)` | Runs one optimizer step over the parameters of the given `layers` using gradients from `tape` for the precomputed scalar `loss`. |
| `Train(Tensor<>,Tensor<>)` | Trains the TabSyn network using the provided input and expected output. |
| `TrainDiffusionBatch(Matrix<>,Int32,Int32,)` | Trains the latent diffusion model on a batch of latent codes. |
| `TrainVAEBatch(Matrix<>,Int32,Int32,)` | Trains the VAE on a batch of rows: forward encode, reparameterize, decode, backward. |
| `UpdateParameters(Vector<>)` |  |
| `ValidateFitInputs(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Validates inputs to the Fit method. |

