---
title: "FlagDiTPredictor<T>"
description: "Flag-DiT noise predictor for the Lumina-T2X image-generation architecture (Gao et al."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.NoisePredictors`

Flag-DiT noise predictor for the Lumina-T2X image-generation architecture
(Gao et al. 2024, "Lumina-T2X: Transforming Text into Any Modality via Flow-based Large
Diffusion Transformers", arXiv:2405.05945).

## For Beginners

Flag-DiT is the transformer behind Lumina. It cuts the image latent into
little square patches (like words), uses rotary attention so it can run at any resolution, and
conditions every layer on the timestep + text via lightweight scale/shift "knobs" that start at
zero (so training begins from a clean identity). Sandwich normalization (a norm on both sides of
each sub-layer) keeps the very deep stack numerically stable.

## How It Works

Faithful Flag-DiT block stack (paper §3.1–3.2): the noisy latent is patchified into a token
sequence, embedded, and processed by N transformer blocks. Each Flag-DiT block uses
**sandwich normalization** (RMSNorm before AND after each sub-layer — Gao 2024 §3.1),
**grouped-query self-attention** with **rotary position embeddings (RoPE)**, a SwiGLU/GELU
feed-forward, and **zero-initialised adaLN** conditioning (Peebles & Xie 2022; the adaLN
projection produces per-block shift/scale/gate from the combined time + text embedding, and is
zero-initialised so every block starts as identity). The final layer applies adaLN + RMSNorm and
projects back to patch space, then unpatchifies to the latent shape. Designed for
rectified-flow training (Lumina-T2X uses a flow-matching scheduler).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FlagDiTPredictor(Int32,Int32,Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new Flag-DiT predictor. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseChannels` |  |
| `ContextDimension` |  |
| `InputChannels` |  |
| `OutputChannels` |  |
| `ParameterCount` |  |
| `SupportsCFG` |  |
| `SupportsCrossAttention` |  |
| `TimeEmbeddingDim` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddWithGate(Tensor<>,Tensor<>,Tensor<>)` | Gated residual: x + gate · branch (gate is a [B,1,hidden] adaLN view). |
| `ApplyAdaLN(Tensor<>,Tensor<>,Tensor<>)` | adaLN modulation: x · (1 + scale) + shift, with [B,1,hidden] broadcast views. |
| `Clone` |  |
| `DeepCopy` |  |
| `FlagDiTLayerSequence` | The full layer list in the canonical (stable) order used by GetParameters/SetParameters. |
| `ForwardBlock(Tensor<>,Tensor<>,Int32)` | One Flag-DiT block: sandwich-normed GQA(+RoPE) and FFN with zero-init adaLN conditioning. |
| `GetParameterChunks` |  |
| `GetParameters` |  |
| `GetTimestepEmbedding(Int32,Int32)` | Sinusoidal timestep embedding (Vaswani 2017 / DDPM), [B, hidden]. |
| `Patchify(Tensor<>)` | [B,C,H,W] -> [B, (H/p)(W/p), C·p·p] via reshape + permute + reshape (tape-tracked). |
| `PoolContext(Tensor<>)` | Mean-pools a context tensor to [B, hidden] after projecting from contextDim. |
| `PredictNoise(Tensor<>,Int32,Tensor<>)` |  |
| `PredictNoiseWithEmbedding(Tensor<>,Tensor<>,Tensor<>)` |  |
| `SetParameterChunks(IEnumerable<Tensor<>>)` |  |
| `SetParameters(Vector<>)` |  |
| `Unpatchify(Tensor<>,Int32,Int32)` | Inverse of `Tensor{`: [B, numPatches, C·p·p] -> [B, C, H, W]. |

## Fields

| Field | Summary |
|:-----|:--------|
| `PatchSize` | Patch size (p): a p×p block of the latent becomes one token (paper uses 2). |

