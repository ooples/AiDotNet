---
title: "DiTNoisePredictor<T>"
description: "Diffusion Transformer (DiT) noise predictor for diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Diffusion Transformer (DiT) noise predictor for diffusion models.

## For Beginners

DiT is the "new generation" of noise prediction:

Traditional U-Net approach:

- Uses convolutional neural networks
- Has encoder-decoder structure with skip connections
- Good, but limited scalability

DiT approach (this class):

- Uses transformer architecture (like GPT, but for images)
- Treats image as patches (like words in a sentence)
- Scales better with more compute and data
- Powers cutting-edge models like DALL-E 3, Sora

Key advantages:

- Better quality at large scales
- Simpler architecture (no skip connections needed)
- More flexible conditioning mechanisms
- Easier to scale training

## How It Works

DiT (Diffusion Transformer) replaces the traditional U-Net architecture with
a pure transformer design. This approach leverages the scalability and
effectiveness of transformers, enabling better performance at larger scales.

Architecture details:

- Patchify: Split image into 2x2 or larger patches
- Position embedding: Add spatial information
- Transformer blocks: Self-attention + MLP
- AdaLN: Adaptive layer normalization for timestep/conditioning
- Unpatchify: Reconstruct full resolution output

Used in: DiT (original), DALL-E 3, Sora, SD3, Pixart-alpha

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiTNoisePredictor(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Double,Int32,Int32,List<DiTNoisePredictor<>.DiTBlock>,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the DiTNoisePredictor class with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AreLayersInitialized` | Whether this predictor's lazy weight tensors have been materialized yet (i.e. |
| `BaseChannels` |  |
| `ContextDimension` |  |
| `HiddenSize` | Gets the hidden size. |
| `InputChannels` |  |
| `NumLayers` | Gets the number of layers. |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `PatchSize` | Gets the patch size. |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `TimeEmbeddingDim` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddPositionEmbedding(Tensor<>,Int32)` | Adds learnable position embeddings using IEngine broadcast add. |
| `AddWithGate(Tensor<>,Tensor<>,Tensor<>)` | Gated residual add: `result = x + gateView * residual`. |
| `ApplyAdaLN(Tensor<>,Tensor<>,Tensor<>)` | Applies adaptive layer normalization (Peebles & Xie, 2023): `y = x * (1 + scale) + shift`. |
| `ApplyCrossAttention(Tensor<>,Tensor<>,DiTNoisePredictor<>.DiTBlock)` | Applies cross-attention between query (from x) and key/value (from conditioning). |
| `ApplySelfAttention(Tensor<>,SelfAttentionLayer<>)` | Applies self-attention. |
| `BuildProbeConditioning` | Builds a representative conditioning tensor for the `Clone` probe forward, matching whichever conditioned path the source has materialized so that path is allocated on the clone before `DiTNoisePredictor{` runs. |
| `Clone` |  |
| `CopyParametersFrom(DiTNoisePredictor<>)` | Copies parameters layer-by-layer from `source` into this predictor. |
| `CreateAttentionLayer` | Creates an attention layer for the transformer block. |
| `CreateDefaultBlocks(Int32)` | Creates industry-standard DiT transformer blocks. |
| `CreatePositionEmbedding(Int32)` | Creates sinusoidal position embeddings. |
| `DeepCopy` |  |
| `EmbedPatches(Tensor<>)` | Embeds patches through linear projection using batched forward pass. |
| `EnsureLayersInitialized` | Ensures layers are initialized (lazy init on first use). |
| `FinalLayerWithAdaLN(Tensor<>,Tensor<>)` | Final layer with AdaLN-zero using batched forward pass. |
| `Forward(Tensor<>,Tensor<>,Tensor<>)` | Forward pass through the DiT. |
| `ForwardBlock(Tensor<>,Tensor<>,Tensor<>,DiTNoisePredictor<>.DiTBlock)` | Forward pass through a single DiT block with AdaLN. |
| `GetParameterChunks` | Streams DiT's materialized trainable tensors directly from each layer, matching the PyTorch `nn.Module.parameters()` contract: yielded tensors are the same objects used by forward/training, not flat copies. |
| `GetParameters` |  |
| `HasMaterializedCrossAttention` | True when any transformer block's cross-attention key projection has materialized its weights, i.e. |
| `InitializeLayers(NeuralNetworkArchitecture<>,Int32,List<DiTNoisePredictor<>.DiTBlock>)` | Initializes all layers of the DiT, using custom layers from the user if provided or creating industry-standard layers from the DiT paper. |
| `Patchify(Tensor<>)` | Converts image to patches via a single reshape + permute + reshape. |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `ProjectTimeEmbedding(Tensor<>)` | Projects timestep embedding through MLP. |
| `ReshapeForHeads(Tensor<>,Int32,Int32,Int32,Int32)` | Reshapes tensor from `[batch, seq, hidden]` to `[batch*heads, seq, headDim]` for multi-head attention via a reshape + permute + reshape pipeline (no scalar nested copy loops). |
| `ReshapeFromHeads(Tensor<>,Int32,Int32,Int32,Int32)` | Inverse of `Int32)` — collapses head and batch back together via reshape + permute + reshape. |
| `SetParameters(Vector<>)` |  |
| `Unpatchify(Tensor<>,Int32,Int32)` | Inverse of `Tensor{` — reconstructs the spatial image. |
| `UseLowPrecisionResidentEval` | True when this forward runs the foundation-scale fp16-resident eval path (no active tape, params over the resident threshold). |

## Fields

| Field | Summary |
|:-----|:--------|
| `LowPrecisionResidentThresholdParams` | Parameter-count floor (≈ 1B params ≈ 4 GB fp32 → 2 GB fp16) above which eval keeps the tower's weights fp16-resident. |
| `_adaln_modulation` | AdaLN modulation for final layer. |
| `_architecture` | The neural network architecture configuration, if provided. |
| `_blocks` | Transformer blocks. |
| `_contextDim` | Context dimension for conditioning. |
| `_finalNorm` | Final layer norm. |
| `_hiddenSize` | Hidden dimension size. |
| `_inputChannels` | Input channels (typically 4 for latent diffusion). |
| `_labelEmbed` | Label/class embedding (optional). |
| `_lastInput` | Cached input for backward pass. |
| `_latentSpatialSize` | Latent spatial size (height = width) for computing patch count. |
| `_mlpRatio` | MLP hidden dimension ratio. |
| `_numHeads` | Number of attention heads. |
| `_numLayers` | Number of transformer layers. |
| `_outputProj` | Output projection (unpatchify). |
| `_patchEmbed` | Patch embedding layer. |
| `_patchSize` | Patch size for image tokenization. |
| `_posEmbed` | Position embeddings (learnable). |
| `_timeEmbed1` | Time embedding layers. |

