---
title: "TransferBenchmark<T>"
description: "Transfer learning benchmark for evaluating SSL representations on downstream tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Evaluation`

Transfer learning benchmark for evaluating SSL representations on downstream tasks.

## For Beginners

Transfer learning benchmarks test how well pretrained
representations transfer to new tasks. We take an encoder pretrained on one dataset
(e.g., ImageNet) and evaluate it on different tasks (e.g., object detection, segmentation).

## How It Works

**Common transfer benchmarks:**

**Evaluation protocols:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TransferBenchmark(INeuralNetwork<>,Int32)` | Initializes a new instance of the TransferBenchmark class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FewShotEvaluation(Tensor<>,Int32[],Tensor<>,Int32[],Int32,Double[])` | Runs few-shot evaluation with limited labeled data. |
| `KNNEvaluation(Tensor<>,Int32[],Tensor<>,Int32[],Int32)` | Runs k-NN evaluation on a downstream dataset. |
| `LinearProbing(Tensor<>,Int32[],Tensor<>,Int32[],Int32,Int32)` | Runs linear probing evaluation on a downstream dataset. |
| `RunFullSuite(Tensor<>,Int32[],Tensor<>,Int32[],Int32,String)` | Runs a full benchmark suite. |

