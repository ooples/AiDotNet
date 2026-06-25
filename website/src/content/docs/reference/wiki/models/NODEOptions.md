---
title: "NODEOptions<T>"
description: "Configuration options for NODE (Neural Oblivious Decision Ensembles)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for NODE (Neural Oblivious Decision Ensembles).

## For Beginners

NODE brings the interpretability of decision trees to deep learning:

- **Oblivious trees**: Simpler trees that are faster to evaluate
- **Soft splits**: Instead of hard left/right decisions, gradual transitions
- **End-to-end training**: Trees are trained with gradient descent like neural networks

This makes NODE both interpretable AND trainable with standard deep learning tools.

Example:

## How It Works

NODE combines differentiable oblivious decision trees with neural network training:

1. Oblivious trees: All nodes at the same depth use the same splitting feature
2. Soft splits: Differentiable split decisions using entmax for sparse attention
3. Ensemble: Multiple trees aggregated for the final prediction

Reference: "Neural Oblivious Decision Ensembles for Deep Learning on Tabular Data" (2019)

## Properties

| Property | Summary |
|:-----|:--------|
| `DropoutRate` | Gets or sets the dropout rate. |
| `EntmaxAlpha` | Gets or sets the entmax alpha parameter for sparse attention. |
| `FeatureSelectionDimension` | Gets or sets the hidden dimension for feature selection. |
| `HiddenActivation` | Gets or sets the hidden layer activation function for the MLP head. |
| `HiddenVectorActivation` | Gets or sets the hidden layer vector activation function (alternative to scalar activation). |
| `InitScale` | Gets or sets the initialization scale for tree parameters. |
| `MLPHiddenDimensions` | Gets or sets the hidden dimensions for the optional MLP head. |
| `NumLeaves` | Gets the number of leaf nodes per tree. |
| `NumTrees` | Gets or sets the number of trees in the ensemble. |
| `Temperature` | Gets or sets the temperature for soft tree splits. |
| `TreeDepth` | Gets or sets the depth of each tree. |
| `TreeOutputDimension` | Gets or sets the output dimension for each tree. |
| `UseBatchNorm` | Gets or sets whether to use batch normalization on inputs. |
| `UseFeaturePreprocessing` | Gets or sets whether to use feature preprocessing before trees. |

