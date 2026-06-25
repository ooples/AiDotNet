---
title: "TCDFAlgorithm<T>"
description: "TCDF — Temporal Causal Discovery Framework."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.DeepLearning`

TCDF — Temporal Causal Discovery Framework.

## For Beginners

TCDF uses "attention" (like in language models) to figure out
which past variables matter for predicting each current variable. If the network
"pays attention" to variable X's past when predicting Y, that suggests X causes Y.

## How It Works

TCDF uses attention-based convolutional neural networks to discover temporal causal
relationships. Each variable has a dedicated 1D-CNN that predicts it from all variables'
histories via causal (left-padded) convolutions. Attention weights over the input
channels indicate which variables are causally relevant for predicting the target.

**Algorithm:**

- For each target variable j, create an attention-weighted CNN
- Attention: a[i,j] = softmax over sigmoid of learnable logits
- Causal convolution: predict x_j[t] from {x_i[t-K:t-1] * a[i,j]} for all i
- Train with MSE loss on next-step prediction
- Final graph: threshold attention weights a[i,j]
- Compute OLS weights for edges above threshold

Reference: Nauta et al. (2019), "Causal Discovery with Attention-Based Convolutional
Neural Networks", Machine Learning and Knowledge Extraction.

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |
| `SupportsTimeSeries` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `DiscoverStructureCore(Matrix<>)` |  |
| `StandardiseColumnsLocal(Matrix<>)` | Zero-mean unit-variance column standardisation. |

