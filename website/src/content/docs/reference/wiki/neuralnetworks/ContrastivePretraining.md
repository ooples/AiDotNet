---
title: "ContrastivePretraining<T>"
description: "Contrastive pretraining module for SAINT architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.Tabular`

Contrastive pretraining module for SAINT architecture.

## For Beginners

Contrastive learning is like a "spot the difference" game:

1. Take an original sample
2. Create a corrupted version (swap some feature values with others)
3. Train the model to tell them apart

This helps the model learn which features are important and how they relate,
without needing labels. It's especially useful when you have lots of unlabeled data.

## How It Works

SAINT uses contrastive learning as a self-supervised pretraining objective.
The model learns to distinguish between original samples and corrupted versions,
which helps learn meaningful representations without labels.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContrastivePretraining(Int32,Int32,Int32,Double,Double)` | Initializes contrastive pretraining module. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ParameterCount` | Gets the total parameter count. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeContrastiveLoss(Tensor<>,Tensor<>)` | Computes contrastive loss between original and corrupted embeddings. |
| `ComputeDenoisingLoss(Tensor<>,Tensor<>)` | Computes denoising loss for reconstructing corrupted features. |
| `CorruptSamples(Tensor<>)` | Creates corrupted samples for contrastive learning. |
| `GetCorruptionMask(Int32)` | Gets the corruption mask indicating which features were corrupted. |
| `ResetState` | Resets internal state. |
| `UpdateParameters(,Tensor<>,Tensor<>)` | Updates parameters. |

