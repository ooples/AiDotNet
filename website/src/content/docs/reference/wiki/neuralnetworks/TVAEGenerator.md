---
title: "TVAEGenerator<T>"
description: "Tabular Variational Autoencoder (TVAE) for generating synthetic tabular data."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Tabular Variational Autoencoder (TVAE) for generating synthetic tabular data.

## For Beginners

TVAE works like a compression algorithm that can also generate new data:

**Training (learning to compress and decompress):****Generation (creating new data):**

The key insight is that the latent space is regularized to be Gaussian,
so we can sample random points from it and decode them into realistic rows.

If you provide custom layers in the architecture, those will be used directly
for the encoder network. If not, the network creates industry-standard
TVAE layers based on the original research paper specifications.

Example usage:

## How It Works

TVAE applies the VAE framework to tabular data with the same VGM preprocessing as CTGAN:

- **Encoder**: Maps transformed data row to latent distribution parameters (mean, logvar)
- **Reparameterization**: z = mean + exp(0.5 * logvar) * epsilon (epsilon ~ N(0,1))
- **Decoder**: Reconstructs the transformed data from the latent code
- **ELBO Loss**: Reconstruction loss (per-column cross-entropy/MSE) + KL divergence

This implementation follows the standard neural network architecture pattern with:

- Proper inheritance from NeuralNetworkBase
- Layer-based architecture using ILayer components
- Engine-based tensor operations for CPU/GPU acceleration
- Full autodiff and JIT compilation support

Reference: "Modeling Tabular Data using Conditional GAN" (Xu et al., NeurIPS 2019)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TVAEGenerator` | Initializes a new TVAE generator with the specified architecture. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Columns` |  |
| `IsFitted` |  |
| `Options` | Gets the TVAE-specific options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyOutputActivations(Tensor<>)` | Applies per-column output activations (tanh for continuous, softmax for categorical). |
| `ComputeElboLossTape(Tensor<>,Tensor<>,Tensor<>,Tensor<>)` | Computes the tape-connected negative ELBO: per-column reconstruction loss (tanh + MSE for the continuous normalized value, softmax + cross-entropy for mode indicators and categorical one-hots — the CTGAN/TVAE loss of Xu et al. |
| `CreateNewInstance` |  |
| `DecoderForward(Tensor<>)` | Decoder forward pass: reconstructs data from latent code z. |
| `DeriveShapeWithLastDim(Int32[],Int32)` | Derives a tensor shape that preserves the rank of the reference but replaces the last dimension. |
| `DeserializeNetworkSpecificData(BinaryReader)` |  |
| `ElboStep(Tensor<>)` | Runs one tape-connected ELBO optimization step on a single transformed input row: encode → reparameterize → decode, then compute the evidence lower bound (per-column reconstruction loss + KL divergence) as a tape-tracked scalar and let auto… |
| `EncoderForward(Tensor<>)` | Encoder forward pass: produces mean and logvar for the latent distribution. |
| `EnsureSizedForInput(Tensor<>)` | When the generator has not yet been fitted (e.g. |
| `ExtractLayerReferences(Int32,Boolean,Int32)` | Re-binds `_encoderLayers`, `_meanLayer`, `_logVarLayer`, and `_decoderLayers` from the shared `Layers` collection using the serialized split, after deserialization replaced Layers with fresh instances. |
| `Fit(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32)` | Fits the TVAE generator to the provided real tabular data. |
| `FitAsync(Matrix<>,IReadOnlyList<ColumnMetadata>,Int32,CancellationToken)` |  |
| `Generate(Int32,Vector<>,Vector<>)` | Generates synthetic tabular data samples. |
| `GetFeatureImportance` |  |
| `GetLayerOutputSize(ILayer<>)` | Gets the output size of a layer by examining its output shape. |
| `GetModelMetadata` |  |
| `InitializeLayers` | Initializes the layers of the TVAE network based on the provided architecture. |
| `PredictCore(Tensor<>)` | Runs the encoder forward pass to produce latent distribution parameters. |
| `RebuildLayersWithActualDimensions(Int32)` | Rebuilds encoder and decoder layers with actual data dimensions discovered during Fit(). |
| `ReduceToScalar(Tensor<>)` | Reduces a tensor to a scalar [1] by summing all elements (tape-connected). |
| `RegisterAllLayers` | Registers every trainable sub-network (encoder, mean/logvar heads, decoder) in the shared `Layers` collection in a stable order so the tape-based training parameter collection and GetParameters/GetParameterGradients/UpdateParameters see the… |
| `Reparameterize(Tensor<>,Tensor<>)` | Reparameterization trick: z = mean + exp(0.5 * logVar) ⊙ epsilon, computed with tape-connected `Engine` ops (epsilon is a sampled constant leaf) so gradients flow into mean and logVar. |
| `SerializeNetworkSpecificData(BinaryWriter)` |  |
| `SoftmaxCrossEntropy(Tensor<>,Tensor<>,Int32,Int32)` | Tape-connected softmax cross-entropy −Σ target·log(softmax(raw_slice)) over a column slice. |
| `SplitEncoderOutput(Tensor<>)` | Splits the encoder output tensor into mean and logvar halves, using tape-connected `Engine` slices so the reparameterization gradient flows back into the encoder. |
| `Train(Tensor<>,Tensor<>)` | Trains the TVAE on a single sample by running one tape-connected ELBO optimization step (encode → reparameterize → decode → reconstruction + KL). |
| `UpdateParameters(Vector<>)` |  |

