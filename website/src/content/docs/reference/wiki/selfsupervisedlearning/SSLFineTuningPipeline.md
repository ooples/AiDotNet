---
title: "SSLFineTuningPipeline<T>"
description: "Pipeline for fine-tuning SSL pretrained encoders on downstream tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning`

Pipeline for fine-tuning SSL pretrained encoders on downstream tasks.

## For Beginners

After SSL pretraining, you typically want to fine-tune
the encoder on a specific task with labeled data. This pipeline handles the
fine-tuning process with proper learning rate schedules and evaluation.

## How It Works

**Fine-tuning strategies:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SSLFineTuningPipeline(INeuralNetwork<>,Int32,Int32)` | Initializes a new fine-tuning pipeline. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Tensor<>,Int32[])` | Evaluates the model on test data. |
| `WithConfig(Action<FineTuningConfig>)` | Configures fine-tuning parameters. |
| `WithStrategy(FineTuningStrategy)` | Sets the fine-tuning strategy. |

## Events

| Event | Summary |
|:-----|:--------|
| `OnProgress` | Event raised for progress updates. |

