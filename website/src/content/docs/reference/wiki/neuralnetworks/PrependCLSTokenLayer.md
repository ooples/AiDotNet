---
title: "PrependCLSTokenLayer<T>"
description: "Prepends a learnable `[CLS]` token to a sequence-of-embeddings input, as introduced by BERT (Devlin et al."
section: "API Reference"
---

`Layers` · `AiDotNet.NeuralNetworks.Layers`

Prepends a learnable `[CLS]` token to a sequence-of-embeddings
input, as introduced by BERT (Devlin et al. 2018 §3.1) and adopted by
ViT / AST (Dosovitskiy et al. 2020 §3.1; Gong et al. 2021 §2.2). The
CLS token starts as a learnable parameter `[1, embedDim]`; the
layer broadcasts it across the batch and concatenates it at sequence
position 0 so the transformer's first output position becomes the
classification representation.

## For Beginners

Most transformer classifiers prepend a
special learnable token to the input sequence; the network learns to
use that one token as a "summary slot" for the whole sequence. After
the transformer runs, you read just that one position to get the
classification feature — no mean-pooling required, and gradient flow
during training teaches the CLS token to aggregate task-relevant
information from the rest of the sequence.

## How It Works

Pairs with `SequenceTokenSliceLayer` using
`First` after the
transformer stack to extract the trained classification embedding —
the canonical AST / ViT classification head.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PrependCLSTokenLayer(Int32,Double,Nullable<Int32>)` | Creates a CLS-token prepender for embedDim-wide inputs. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` |  |
| `SupportsTraining` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Forward(Tensor<>)` |  |
| `GetMetadata` |  |
| `GetParameters` |  |
| `GetTrainableParameters` |  |
| `ResetState` |  |
| `SetParameters(Vector<>)` |  |
| `SetTrainableParameters(IReadOnlyList<Tensor<>>)` |  |
| `UpdateParameters()` |  |

