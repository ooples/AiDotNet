---
title: "CausalGANOptions<T>"
description: "Configuration options for Causal-GAN, a GAN that learns causal graph structure and generates data respecting causal relationships between features."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Causal-GAN, a GAN that learns causal graph structure
and generates data respecting causal relationships between features.

## For Beginners

Causal-GAN learns which features cause other features:

Instead of just learning correlations (e.g., "Age and Income are related"),
it learns causation (e.g., "Education causes higher Income").

This allows:

1. Generating more realistic data (respecting cause-effect chains)
2. Answering "what if" questions ("what if everyone had a college degree?")

Example:

## How It Works

Causal-GAN discovers and uses causal structure:

- **DAG structure learning**: NOTEARS-style continuous relaxation for learning a directed acyclic graph
- **Structural equation models**: Each feature is generated as a function of its causal parents
- **Interventional generation**: Can simulate "what-if" scenarios by intervening on specific features

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the training batch size. |
| `DAGPenaltyWeight` | Gets or sets the weight for the DAG acyclicity penalty (NOTEARS constraint). |
| `DiscriminatorDropout` | Gets or sets the dropout rate for discriminator hidden layers. |
| `DiscriminatorSteps` | Gets or sets the number of discriminator training steps per generator step. |
| `EmbeddingDimension` | Gets or sets the noise dimension. |
| `Epochs` | Gets or sets the number of training epochs. |
| `GradientPenaltyWeight` | Gets or sets the weight for the WGAN-GP gradient penalty term. |
| `HiddenDimensions` | Gets or sets the hidden layer sizes. |
| `LearningRate` | Gets or sets the learning rate. |
| `SparsityWeight` | Gets or sets the sparsity weight for the learned adjacency matrix. |
| `VGMModes` | Gets or sets the number of VGM modes for continuous column encoding. |

