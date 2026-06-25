---
title: "ProgressiveDistillationTrainer<T>"
description: "Trainer for progressive distillation that halves the number of steps in each round."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Trainer for progressive distillation that halves the number of steps in each round.

## For Beginners

Imagine learning to draw in fewer strokes. First, you learn to
do in 512 strokes what took 1024. Then 256, then 128, and so on. Each time you learn
to combine two teacher steps into one student step. After several rounds, you can
draw the same picture in just 4-8 strokes.

## How It Works

Progressive distillation trains a student to match the output of a teacher's two steps
in a single step. Applied iteratively: 1024→512→256→...→4→2→1 steps. Each round halves
the step count while maintaining quality, producing a model that generates in few steps.

Reference: Salimans and Ho, "Progressive Distillation for Fast Sampling of Diffusion Models", ICLR 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProgressiveDistillationTrainer(IDiffusionModel<>,Int32,Int32)` | Initializes a new progressive distillation trainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InitialSteps` | Gets the initial step count. |
| `NumRounds` | Gets the number of distillation rounds needed. |
| `TargetSteps` | Gets the target step count. |
| `Teacher` | Gets the teacher model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistillationLoss(Vector<>,Vector<>)` | Computes the progressive distillation loss (student one-step vs teacher two-step). |

