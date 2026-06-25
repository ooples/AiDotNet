---
title: "TransformerDecoderBlock<T>"
description: "Pre-Layer-Normalization transformer decoder block — self-attention, a second (\"cross\") attention, and a position-wise feed-forward network, each wrapped in a residual (skip) connection with layer normalization applied BEFORE the sublayer."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Pre-Layer-Normalization transformer decoder block — self-attention, a second
("cross") attention, and a position-wise feed-forward network, each wrapped in a
residual (skip) connection with layer normalization applied BEFORE the sublayer.

## How It Works

Block structure (Pre-LN; residual/FFN design Vaswani 2017 §3.1):

As with `TransformerEncoderBlock`, the **residual connections**
are what let the input signal flow through depth — their absence was the root
cause of issue #1380's mode-collapse. This block restores them for the decoder
stack. NOTE: in this sequential model the "cross" attention attends over the
decoder's own hidden stream (there is no separate encoder-memory input threaded
through the layer list); that pre-existing limitation is unchanged here — this
block only adds the missing residual connections and Pre-LN ordering.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransformerDecoderBlock(Int32,Int32,Int32,Double)` | Initialises a Pre-LN transformer decoder block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CrossAttentionLayer` | The block's cross-attention sublayer. |
| `DropoutRate` | Dropout probability — persisted for deserialization. |
| `FfnDim` | Feed-forward inner dimension — persisted for deserialization. |
| `FfnDownLayer` | The block's current FFN down-projection (ffnDim → hiddenSize) sublayer. |
| `FfnUpLayer` | The block's current FFN up-projection (hiddenSize → ffnDim) sublayer. |
| `HiddenSize` | Model (feature) dimension — persisted for deserialization. |
| `NumHeads` | Number of attention heads — persisted for deserialization. |
| `ParameterCount` |  |
| `SelfAttentionLayer` | The block's current self-attention sublayer (replaceable via `LayerBase{`). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearGradients` |  |
| `EnsureInitialized` | Auto-generated EnsureInitialized: registers sub-layers (cheap), then delegates to base for weight allocation. |
| `EnsureSubLayersRegistered` | Registers discovered sub-layer fields exactly once. |
| `Forward(Tensor<>)` | Pre-LN forward pass WITHOUT an encoder context; all ops route through Engine/sublayers so the tape records them. |
| `Forward(Tensor<>,Tensor<>)` | Pre-LN forward pass with a true encoder-decoder cross-attention sublayer: the cross-attention queries come from the decoder stream and the keys/values from `encoderOutput` (Vaswani 2017 §3.2.3). |
| `Forward(Tensor<>[])` | Multi-input dispatch: 1 input = decoder-only (degenerate cross-attention), 2 inputs = (decoderStream, encoderOutput) true cross-attention. |
| `GetMetadata` | Persists ctor params for deserialization (no (int[] inputShape) ctor). |
| `GetParameterGradients` |  |
| `GetParameters` |  |
| `ReplaceFfnDown(LayerBase<>)` | Swaps the FFN down-projection sublayer. |
| `ReplaceFfnUp(LayerBase<>)` | Swaps the FFN up-projection sublayer. |
| `ReplaceSelfAttention(LayerBase<>)` | Swaps the self-attention sublayer (e.g. |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainingMode(Boolean)` |  |
| `UpdateParameters()` |  |

