---
title: "TeacherStudentSSL<T>"
description: "Base class for teacher-student self-supervised learning methods."
section: "API Reference"
---

`Base Classes` · `AiDotNet.SelfSupervisedLearning`

Base class for teacher-student self-supervised learning methods.

## For Beginners

Teacher-student SSL methods use two networks:
a student that learns from gradients and a teacher that provides targets.
The teacher is typically updated as an exponential moving average (EMA)
of the student, providing stable learning targets.

## How It Works

**Common components:**

**Methods using this pattern:** DINO, iBOT, EsViT, DINOv2

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TeacherStudentSSL(INeuralNetwork<>,IMomentumEncoder<>,IProjectorHead<>,IProjectorHead<>,Int32,SSLConfig)` | Initializes a new instance of the TeacherStudentSSL class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumGlobalCrops` | Number of global crops (larger views used by both student and teacher). |
| `NumLocalCrops` | Number of local crops (smaller views used by student only). |
| `RequiresMemoryBank` |  |
| `UsesMomentumEncoder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateMultiCropViews(Tensor<>)` | Creates augmented views for teacher-student training. |
| `ForwardStudent(Tensor<>)` | Performs forward pass through student network. |
| `ForwardTeacher(Tensor<>)` | Performs forward pass through teacher network (no gradients). |
| `GetAdditionalParameterCount` |  |
| `GetAdditionalParameters` |  |
| `OnEpochStart(Int32)` |  |
| `UpdateStudent()` | Updates student network parameters with gradients. |
| `UpdateTeacher` | Updates teacher network with EMA from student. |

## Fields

| Field | Summary |
|:-----|:--------|
| `Augmentation` | Augmentation policies for creating views. |
| `BaseMomentum` | Base momentum value for teacher updates. |
| `Centering` | Centering mechanism to prevent collapse. |
| `TeacherEncoder` | The teacher encoder (momentum-updated copy of student). |
| `TeacherProjector` | The teacher projection head. |

