---
title: "HyperNetMetaRLOptions<T, TInput, TOutput>"
description: "Configuration options for HyperNet Meta-RL: hypernetwork-based policy generation for meta-reinforcement learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for HyperNet Meta-RL: hypernetwork-based policy generation
for meta-reinforcement learning.

## How It Works

HyperNet Meta-RL uses a hypernetwork to generate task-specific policy parameters
from a task embedding. The task embedding is computed from initial task interaction
data (support set), and the hypernetwork transforms it into a full parameter
vector for the policy network, enabling single-forward-pass adaptation.

## Properties

| Property | Summary |
|:-----|:--------|
| `HyperNetHiddenDim` | Hidden dimension of the hypernetwork. |
| `ParamRegWeight` | Weight for the parameter regularization loss. |
| `TaskEmbeddingDim` | Dimensionality of the task embedding. |

