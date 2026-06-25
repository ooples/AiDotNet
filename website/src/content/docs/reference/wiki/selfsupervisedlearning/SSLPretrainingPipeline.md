---
title: "SSLPretrainingPipeline<T>"
description: "High-level pipeline for SSL pretraining."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

High-level pipeline for SSL pretraining.

## For Beginners

This pipeline provides a simple, high-level interface
for SSL pretraining. Just provide your encoder and data, and it handles the
rest: method selection, augmentation, training loop, and evaluation.

## How It Works

**Example usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSLPretrainingPipeline(INeuralNetwork<>,Int32)` | Initializes a new SSL pretraining pipeline. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Train(Func<IEnumerable<Tensor<>>>,Tensor<>,Int32[])` | Trains the encoder using SSL. |
| `WithConfig(Action<SSLConfig>)` | Configures the training parameters. |
| `WithEncoderCopyFactory(Func<INeuralNetwork<>,INeuralNetwork<>>)` | Sets the function to create encoder copies (for momentum methods). |
| `WithMethod(SSLMethodType)` | Sets the SSL method to use. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnProgress` | Event raised during training for progress updates. |

