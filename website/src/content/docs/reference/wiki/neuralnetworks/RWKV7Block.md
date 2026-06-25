---
title: "RWKV7Block<T>"
description: "Implements a single RWKV-7 \"Goose\" block with the WKV-7 kernel featuring dynamic state evolution."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers.SSM`

Implements a single RWKV-7 "Goose" block with the WKV-7 kernel featuring dynamic state evolution.

## For Beginners

This is one layer in the RWKV-7 model. Think of it as a smart
information processor that:

1. Reads the current word and blends it with the previous word
2. Decides what to remember and what to forget (using learnable transition rules)
3. Produces an output that captures both local and long-range context

Unlike Transformers that re-read the entire text each time, RWKV-7 keeps a compact running
summary (the "state") that gets updated with each new word, making it very efficient.

## How It Works

RWKV-7 is the seventh generation of the RWKV architecture, introducing expressive dynamic state
evolution that replaces the fixed exponential decay of previous versions with learnable, data-dependent
transition matrices. This allows the model to dynamically control how information is stored, retained,
and forgotten in the recurrent state.

Each block contains two sub-layers with residual connections:

Key innovations over RWKV-6 "Finch":

- Learnable transition vectors a_t (diagonal state decay) and b_t (additive state injection)
- State evolution: S_t = diag(a_t) * S_{t-1} + b_t * (k_t * v_t), replacing fixed exp decay
- Group normalization on WKV output for stability
- SiLU activation in channel mixing instead of squared ReLU

**Reference:** Peng et al., "RWKV-7 Goose with Expressive Dynamic State Evolution", 2025.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RWKV7Block(Int32,Int32,Int32,Double,IActivationFunction<>,IInitializationStrategy<>)` | Creates a new RWKV-7 block. |

## Properties

| Property | Summary |
|:-----|:--------|
| `FFNDimension` | Gets the feed-forward network dimension. |
| `HeadDimension` | Gets the dimension per head. |
| `ModelDimension` | Gets the model dimension. |
| `NumHeads` | Gets the number of attention heads. |
| `ParameterCount` |  |
| `SupportsTraining` | Training support is approximate: gradients flow through residual connections and weight gradients are accumulated, but full backpropagation through the WKV-7 recurrent kernel is not yet implemented. |
| `Ws` | Gets the workspace, throwing if not initialized. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AccumulateChannelMixGradients(Tensor<>,Tensor<>,Int32,Int32)` | Accumulates gradients for the channel mixing sub-layer. |
| `AccumulateTimeMixGradients(Tensor<>,Tensor<>,Int32,Int32)` | Accumulates gradients for the output projection weights from the time mixing sub-layer. |
| `AccumulateTimeMixParameterGradients(Tensor<>,Tensor<>,Int32,Int32)` | Computes gradients for all time mixing parameters: timeMixR/K/V/A/B, receptance/key/value weights, a/b weights+biases, groupNorm gamma/beta. |
| `ApplyGroupNorm(Tensor<>,Int32)` | Applies group normalization across heads. |
| `ApplyLayerNorm(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Applies layer normalization. |
| `ChannelMixingForward(Tensor<>,Int32,Int32)` | Channel mixing forward with SiLU gating (RWKV-7 style). |
| `ClearGradients` |  |
| `ComputeFinalWkvState(Tensor<>,Tensor<>,Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | Computes the final WKV recurrent state S_T after processing a sequence, for autoregressive streaming inference. |
| `Forward(Tensor<>)` |  |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetPreviousToken` | Gets the previous token for token-shift continuation. |
| `GetRecurrentState` | Gets the recurrent state for autoregressive continuation. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `GroupNormBackward(Tensor<>,Tensor<>,Int32)` | GroupNorm backward pass. |
| `LayerNormBackward(Tensor<>,Tensor<>,Tensor<>,Int32,Int32)` | LayerNorm backward: returns gradient w.r.t. |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SafeSetSlice(Tensor<>,Int32,Tensor<>,Int32,Int32)` | Accumulates gradients for LayerNorm gamma and beta parameters. |
| `SetParameters(Vector<>)` |  |
| `SetPreviousToken(Tensor<>)` | Sets the previous token for token-shift continuation. |
| `SetRecurrentState(Tensor<>)` | Sets the recurrent state for autoregressive continuation. |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `TimeMixingForward(Tensor<>,Int32,Int32)` | Time mixing forward with WKV-7 dynamic state evolution kernel. |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

