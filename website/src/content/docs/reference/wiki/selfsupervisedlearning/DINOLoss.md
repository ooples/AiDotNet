---
title: "DINOLoss<T>"
description: "DINO (Self-Distillation with No Labels) Loss for self-supervised learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.SelfSupervisedLearning.Losses`

DINO (Self-Distillation with No Labels) Loss for self-supervised learning.

## For Beginners

DINO loss is a cross-entropy loss between student and teacher
outputs, where the teacher is an EMA of the student. It uses centering and sharpening
to prevent collapse.

## How It Works

**Key components:**

**Loss formula:**

**Reference:** Caron et al., "Emerging Properties in Self-Supervised Vision
Transformers" (ICCV 2021)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DINOLoss(Int32,Double,Double,Double)` | Initializes a new instance of the DINOLoss class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `StudentTemperature` | Gets the student temperature parameter. |
| `TeacherTemperature` | Gets the teacher temperature parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AiDotNet#Interfaces#IContrastiveLoss{T}#ComputeLoss(Tensor<>,Tensor<>)` | IContrastiveLoss implementation — delegates to ComputeLoss with default center update. |
| `ComputeLoss(Tensor<>,Tensor<>,Boolean)` | Computes the DINO loss between student and teacher outputs. |
| `ComputeLossWithGradients(Tensor<>,Tensor<>)` | Computes DINO loss with gradients for backpropagation. |
| `ComputeMultiCropLoss(IList<Tensor<>>,IList<Tensor<>>)` | Computes DINO loss for multiple student crops against global teacher views. |
| `GetCenter` | Gets the current center values. |
| `ResetCenter` | Resets the center to zeros. |

