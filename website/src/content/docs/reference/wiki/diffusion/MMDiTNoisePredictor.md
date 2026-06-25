---
title: "MMDiTNoisePredictor<T>"
description: "Multi-Modal Diffusion Transformer (MMDiT) noise predictor for SD3 and FLUX architectures."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Multi-Modal Diffusion Transformer (MMDiT) noise predictor for SD3 and FLUX architectures.

## For Beginners

MMDiT is the architecture behind SD3 and FLUX:

How MMDiT differs from standard DiT:

- Standard DiT: Image patches processed by transformer, text injected via cross-attention
- MMDiT: Image AND text tokens are concatenated and processed together through JOINT attention

Key characteristics:

- Joint attention: Both text and image tokens attend to each other equally
- Separate MLPs: Text and image tokens have independent MLP layers after joint attention
- Dual stream: Text and image streams with shared attention but separate feed-forward
- AdaLN-Zero: Adaptive layer normalization with zero-init gating
- Supports multiple text encoders (CLIP + T5 for SD3, CLIP + T5 for FLUX)

Used in:

- Stable Diffusion 3 / SD 3.5 (Stability AI)
- FLUX.1 dev/schnell/pro (Black Forest Labs)
- Pixart-Sigma

Advantages over standard DiT:

- Better text-image alignment through joint attention
- More expressive conditioning without cross-attention bottleneck
- Scales better with model size

## How It Works

MMDiT extends the standard DiT architecture by processing text and image tokens
jointly through shared transformer blocks, rather than using cross-attention.
This enables deeper bidirectional interaction between modalities.

Technical specifications:

- Architecture: Multi-modal transformer with joint self-attention
- SD3 Medium: 2B params, 24 layers, hidden 1536, 24 heads
- FLUX.1 dev: ~12B params, 19 double + 38 single layers, hidden 3072, 24 heads
- Patch size: 2 (latent space)
- Conditioning: Concatenated CLIP + T5 text embeddings
- Timestep: Sinusoidal + MLP projection
- Positional encoding: RoPE (Rotary Position Embedding)

Reference: Esser et al., "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis", ICML 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `MMDiTNoisePredictor(NeuralNetworkArchitecture<>,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,List<MMDiTNoisePredictor<>.MMDiTBlock>,List<MMDiTNoisePredictor<>.MMDiTSingleBlock>,ILossFunction<>,Nullable<Int32>)` | Initializes a new instance of the MMDiTNoisePredictor class with full customization support. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` |  |
| `ContextDimension` |  |
| `HiddenSize` | Gets the hidden size of the transformer. |
| `InputChannels` |  |
| `NumJointLayers` | Gets the number of joint (double-stream) transformer layers. |
| `NumSingleLayers` | Gets the number of single-stream transformer layers (FLUX-style). |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `PatchSize` | Gets the patch size for latent tokenization. |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `TimeEmbeddingDim` |  |
| `WeightsMaterialized` | True once this predictor's lazy weights have been materialized (its patch-embed is initialized, which the first forward triggers). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `CopyParametersFrom(MMDiTNoisePredictor<>)` | Copies trained weights from `source` into this predictor layer by layer, without ever materializing a single contiguous parameter vector. |
| `DeepCopy` |  |
| `EnumerateLayers` | Yields only the layers this predictor owns so `Boolean)` tears down their pool-rented weight tensors. |
| `ForwardJointBlock(Tensor<>,Tensor<>,Tensor<>,MMDiTNoisePredictor<>.MMDiTBlock)` | Processes a joint (double-stream) block where text and image attend to each other. |
| `ForwardSingleBlock(Tensor<>,Tensor<>,MMDiTNoisePredictor<>.MMDiTSingleBlock)` | Processes a single-stream block (FLUX-style) on concatenated tokens. |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `InitializeLayers(NeuralNetworkArchitecture<>,List<MMDiTNoisePredictor<>.MMDiTBlock>,List<MMDiTNoisePredictor<>.MMDiTSingleBlock>)` | Initializes all layers of the MMDiT, using custom layers from the user if provided or creating industry-standard layers. |
| `MMDiTLayerSequence` | The full layer list in the EXACT order GetParameters/SetParameters serialize it. |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` |  |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `SetParameterChunks(IEnumerable<Tensor<>>)` |  |
| `SetParameters(Vector<>)` |  |
| `ToHeads4D(Tensor<>,Int32,Int32,Int32,Int32)` | [B, seq, H·D] → [B, H, seq, D] for the engine's multi-head SDPA. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_ownsBlocks` | True when this predictor created the joint/single blocks itself (defaults or architecture-driven), false when they were supplied by the caller via customJointBlocks/customSingleBlocks. |

