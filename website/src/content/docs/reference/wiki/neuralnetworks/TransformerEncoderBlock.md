---
title: "TransformerEncoderBlock<T>"
description: "Pre-Layer-Normalization transformer encoder block — multi-head self-attention and a position-wise feed-forward network, each wrapped in a residual (skip) connection with layer normalization applied BEFORE the sublayer (Pre-LN)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Pre-Layer-Normalization transformer encoder block — multi-head self-attention
and a position-wise feed-forward network, each wrapped in a residual (skip)
connection with layer normalization applied BEFORE the sublayer (Pre-LN).

## For Beginners

A residual connection means "add the layer's
input back to its output." It's like keeping a copy of the original so
nothing important gets lost as the data passes through. Transformers
literally cannot learn without them.

## How It Works

Block structure (Pre-LN, Xiong et al. 2020 "On Layer Normalization in the
Transformer Architecture"; the residual/FFN design is Vaswani 2017 §3.1):

The **residual connections** (the `x +` / `y +` terms) are the
defining feature of the transformer: they let the input signal flow
unattenuated through arbitrarily deep stacks. Without them the attention/FFN
output REPLACES the hidden state each layer, the token-identity signal is
washed out (empirically ~60× per layer), and the network mode-collapses to
an input-independent constant output — the root cause of issue #1380.

**Pre-LN vs Post-LN:** normalizing the sublayer INPUT (Pre-LN) rather than
the residual SUM (Post-LN, the original 2017 ordering) keeps the residual
path un-normalized, so gradients flow cleanly through depth and the model
trains stably WITHOUT learning-rate warmup. Post-LN converges far slower
without warmup; Pre-LN is the ordering used by every modern transformer
(GPT-2 onward, LLaMA, etc.) and trains measurably faster here.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerEncoderBlock(Int32,Int32,Int32,Double,IActivationFunction<>)` | Initialises a Post-LN transformer encoder block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AttentionLayer` | The block's current self-attention sublayer. |
| `DropoutRate` | Dropout probability — persisted for deserialization. |
| `FfnDim` | Feed-forward inner dimension — persisted for deserialization. |
| `FfnDownLayer` | The block's current FFN down-projection (ffnDim → hiddenSize) sublayer. |
| `FfnUpLayer` | The block's current FFN up-projection (hiddenSize → ffnDim) sublayer. |
| `HiddenSize` | Model (feature) dimension — persisted for deserialization. |
| `NumHeads` | Number of attention heads — persisted for deserialization. |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` | Forward pass. |
| `GetMetadata` | Persists the constructor's full parameter set so `DeserializationHelper.CreateLayerFromType` can reconstruct the block (it has no `(int[] inputShape)` constructor). |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `MaterializeLazySublayers` | Runs a dummy forward at the known hidden size to force the lazy Dense FFN sublayers to allocate their weights, so `ParameterCount` reflects the full block. |
| `ReplaceAttention(LayerBase<>)` | Swaps the block's self-attention sublayer for `replacement` — the composite-layer counterpart of the inference optimizer assigning a rewritten attention layer into `model.Layers[i]` for discrete layouts. |
| `ReplaceFfnDown(LayerBase<>)` | Swaps the FFN down-projection sublayer (e.g. |
| `ReplaceFfnUp(LayerBase<>)` | Swaps the FFN up-projection sublayer (e.g. |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |
| `UpdateParameters()` |  |

