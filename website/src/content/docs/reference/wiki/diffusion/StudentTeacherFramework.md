---
title: "StudentTeacherFramework<T>"
description: "Generic student-teacher framework for knowledge distillation in diffusion models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Distillation`

Generic student-teacher framework for knowledge distillation in diffusion models.

## For Beginners

This is like a tutoring framework. A knowledgeable teacher (the full
diffusion model) teaches a student (a faster model) to produce similar results. The framework
manages how the teacher shows examples and how the student learns, supporting different
teaching styles (consistency, progressive, adversarial distillation).

## How It Works

Provides the foundational abstraction for all distillation approaches where a smaller/faster
student model learns to replicate the behavior of a larger/slower teacher model. Handles
EMA (Exponential Moving Average) target network updates, timestep sampling strategies,
and loss computation delegation.

Reference: Hinton et al., "Distilling the Knowledge in a Neural Network", NeurIPS Workshop 2015;
adapted for diffusion models by Salimans & Ho, "Progressive Distillation for Fast Sampling
of Diffusion Models", ICLR 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StudentTeacherFramework(IDiffusionModel<>,IDiffusionModel<>,Double,Double)` | Initializes a new student-teacher framework. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EMADecay` | Gets the EMA decay rate for the target network. |
| `Student` | Gets the student diffusion model being trained. |
| `Teacher` | Gets the teacher diffusion model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeDistillationLoss(Vector<>,Vector<>)` | Computes the distillation loss between student and teacher outputs. |
| `SampleTimestep(Int32,Random)` | Samples a random timestep for training. |
| `UpdateEMA(Vector<>)` | Updates the EMA target network parameters from the student's current parameters. |

