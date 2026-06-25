---
title: "ETPNOptions<T, TInput, TOutput>"
description: "Configuration options for ETPN (Embedding-Transformed Prototypical Networks)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for ETPN (Embedding-Transformed Prototypical Networks).

## How It Works

ETPN learns a task-specific embedding transformation applied transductively using
both support and query gradient information. The transformation adapts the parameter
space to be more discriminative for the specific task at hand.

## Properties

| Property | Summary |
|:-----|:--------|
| `QueryInfluenceWeight` | How much query gradient information influences the transform. |
| `TransductiveIterations` | Number of transductive iterations using query feedback. |
| `TransformDim` | Dimension of the embedding transform space. |
| `TransformRegWeight` | Regularization on the learned transform parameters. |

