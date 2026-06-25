---
title: "LinearEvaluator<T>"
description: "Linear evaluation protocol for assessing SSL representation quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Evaluation`

Linear evaluation protocol for assessing SSL representation quality.

## For Beginners

Linear evaluation is the standard way to measure how good
self-supervised representations are. We freeze the pretrained encoder and train only
a simple linear classifier on top. Better representations = higher accuracy.

## How It Works

**Why linear evaluation?**

**Typical protocol:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LinearEvaluator(INeuralNetwork<>,Int32,Int32,Double,Int32)` | Initializes a new instance of the LinearEvaluator class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputDimension` | Gets the input dimension (encoder output). |
| `NumClasses` | Gets the number of classes for classification. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Evaluate(Tensor<>,Int32[])` | Evaluates the linear classifier on a test dataset. |
| `EvaluateTopK(Tensor<>,Int32[],Int32)` | Computes top-k accuracy. |
| `Train(Tensor<>,Int32[],Tensor<>,Int32[])` | Trains the linear classifier on the given dataset. |

