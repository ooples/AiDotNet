---
title: "SSLConfig"
description: "Unified configuration for self-supervised learning with industry-standard defaults."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

Unified configuration for self-supervised learning with industry-standard defaults.

## For Beginners

Self-supervised learning (SSL) learns useful representations
from unlabeled data. This configuration controls how SSL pretraining works, including
the method to use, batch size, training epochs, and method-specific settings.

## How It Works

**Key features:**

**Example - Simple usage with defaults:**

**Example - Custom configuration:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSLConfig` | Creates a new SSL configuration with industry-standard defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BYOL` | Gets or sets BYOL-specific configuration. |
| `BarlowTwins` | Gets or sets Barlow Twins-specific configuration. |
| `BatchSize` | Gets or sets the batch size for pretraining. |
| `CheckpointFrequency` | Gets or sets the checkpoint save frequency (epochs). |
| `CheckpointPath` | Gets or sets the path to save checkpoints. |
| `DINO` | Gets or sets DINO-specific configuration. |
| `Distributed` | Gets or sets the distributed training configuration. |
| `EnableKNNEvaluation` | Gets or sets whether to run k-NN evaluation. |
| `EnableLinearEvaluation` | Gets or sets whether to run linear evaluation during/after pretraining. |
| `KNNNeighbors` | Gets or sets the number of neighbors for k-NN evaluation. |
| `LearningRate` | Gets or sets the base learning rate. |
| `LinearEvaluationFrequency` | Gets or sets the frequency of linear evaluation (epochs). |
| `MAE` | Gets or sets MAE-specific configuration. |
| `Method` | Gets or sets the SSL method to use. |
| `MoCo` | Gets or sets MoCo-specific configuration. |
| `OptimizerType` | Gets or sets the optimizer type for SSL training. |
| `PretrainingEpochs` | Gets or sets the number of pretraining epochs. |
| `ProjectorHiddenDimension` | Gets or sets the hidden dimension of the projection head MLP. |
| `ProjectorLayers` | Gets or sets the number of layers in the projection head. |
| `ProjectorOutputDimension` | Gets or sets the output dimension of the projection head. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `Temperature` | Gets or sets the temperature parameter for contrastive loss. |
| `UseCosineDecay` | Gets or sets whether to use cosine learning rate decay. |
| `UseTemperatureScheduling` | Gets or sets whether to use temperature scheduling. |
| `WarmupEpochs` | Gets or sets the number of warmup epochs. |
| `WeightDecay` | Gets or sets the weight decay (L2 regularization). |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary for logging or serialization. |

