---
title: "OMLOptions<T, TInput, TOutput>"
description: "Configuration options for the OML (Online Meta-Learning) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for the OML (Online Meta-Learning) algorithm.

## How It Works

OML (Javed & White, 2019) partitions the model into a Representation Learning
Network (RLN) and a Prediction Learning Network (PLN). Only the PLN parameters are
adapted in the inner loop, while the RLN is meta-learned to produce sparse,
non-interfering representations that enable continual learning.

## Properties

| Property | Summary |
|:-----|:--------|
| `PlnFraction` | Fraction of total model parameters that form the PLN (prediction head). |
| `RepresentationRegWeight` | L2 regularization weight on RLN parameter changes to prevent catastrophic forgetting of the learned representation. |
| `SparsityPenalty` | L1 sparsity penalty on the RLN parameter activations to encourage non-interfering, sparse representations. |

