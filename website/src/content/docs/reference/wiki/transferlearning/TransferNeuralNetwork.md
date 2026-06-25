---
title: "TransferNeuralNetwork<T>"
description: "Implements transfer learning for Neural Network models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TransferLearning.Algorithms`

Implements transfer learning for Neural Network models.

## For Beginners

This class enables neural networks to transfer knowledge from one
domain to another, even when the feature spaces are different. It uses techniques like
adapter layers and knowledge distillation to make this possible.

## Methods

| Method | Summary |
|:-----|:--------|
| `CombineLabels(Vector<>,Vector<>,Double)` | Combines soft labels from source model with true target labels. |
| `Transfer(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Matrix<>,Vector<>)` | Transfers a Neural Network model to a target domain with proper source data. |
| `TransferCrossDomain(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Vector<>)` | Transfers a Neural Network model to a target domain with a different feature space. |
| `TransferSameDomain(IFullModel<,Matrix<>,Vector<>>,Matrix<>,Vector<>)` | Transfers a Neural Network model to a target domain with the same feature space. |

