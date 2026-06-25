---
title: "T5RelativeBiasAttentionLayer<T>"
description: "T5-style multi-head self-attention with learned relative position bias (Raffel et al., \"Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer\", JMLR 2020)."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

T5-style multi-head self-attention with learned relative position bias
(Raffel et al., "Exploring the Limits of Transfer Learning with a
Unified Text-to-Text Transformer", JMLR 2020).

## For Beginners

Standard attention layers learn position information
only through fixed sinusoidal patterns added to the input. T5 instead
learns directly how much to bias attention between any two positions —
a richer, fully-trained position signal that has been a major contributor
to T5's strong empirical performance.

## How It Works

Two paper-faithful deviations from the standard MultiHeadAttention layer:

**Shared-bias convention:** The original T5 paper shares ONE relative-
position bias table across all encoder layers (Raffel 2020 §2.1 footnote 5,
"the relative position embedding is shared across all layers but each head
has its own embedding"). The constructor accepts an optional
`sharedRelativeBiasTable`; when supplied, this layer reuses it instead
of allocating its own. The `LayerHelper.CreateDefaultT5TextLayers`
factory wires one shared table through every T5 attention layer in the
stack — that is the paper-canonical configuration. Standalone construction
(e.g. unit tests) gives each layer its own bias table, which is the
"common-but-non-canonical" HuggingFace T5 default.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `T5RelativeBiasAttentionLayer(Int32,Int32,Int32,Int32,Boolean,Nullable<Int32>,Tensor<>)` | Initialises a new T5-style relative-bias self-attention layer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `OwnsRelativeBiasTable` | Returns whether this layer owns its bias table (vs. |
| `ParameterCount` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildT5RelativeBias(Int32)` | Builds the T5 relative position bias tensor of shape `[numHeads, seqLen, seqLen]` by looking up the trainable bias table with bucketed relative-position indices. |
| `ClearGradients` |  |
| `ComputeBucketIndices(Int32)` | Computes the bucket index for every (queryPos, keyPos) pair in a sequence of length `seqLen`, following the T5 reference implementation (mesh-tensorflow's `_relative_position_bucket`). |
| `Forward(Tensor<>)` | Performs the forward pass: project Q/K/V, scaled dot-product attention with the T5 relative bias added pre-softmax, then output projection. |
| `GetMetadata` | Returns layer-specific metadata required for cloning / serialisation. |
| `GetParameterGradients` |  |
| `GetParameterRoles` | Returns parameter roles for per-role optimizer configuration (e.g., weight decay exemption for biases). |
| `GetParameters` |  |
| `GetRelativeBiasTable` | Gets the relative bias table for testing and inspection. |
| `GetTrainableParameters` | Returns all trainable parameter tensors marked with [TrainableParameter]. |
| `InitProjection(Int32,Int32)` | Glorot / Xavier-uniform initialisation for a [fanIn, fanOut] weight matrix. |
| `RelativePositionBucket(Int32,Boolean,Int32,Int32)` | Canonical T5 relative-position bucketing (Raffel 2020; mesh-tensorflow reference `_relative_position_bucket`). |
| `ResetState` |  |
| `ReturnPooledParameters` | Returns rented parameter tensors to the TensorAllocator pool. |
| `SampleNormalTensor(Int32[],Double)` | Samples a normal-distributed [shape...] tensor with mean 0 and the requested standard deviation via Box-Muller. |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` | Replaces trainable parameter tensors (e.g., with ParameterBuffer views). |
| `UpdateParameters()` |  |
| `ZeroGrad` | Clears all gradient fields discovered by convention ({paramName}Gradient). |

