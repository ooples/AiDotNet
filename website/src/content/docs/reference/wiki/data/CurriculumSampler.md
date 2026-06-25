---
title: "CurriculumSampler<T>"
description: "A sampler that implements curriculum learning by progressively introducing harder samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Sampling`

A sampler that implements curriculum learning by progressively introducing harder samples.

## For Beginners

Imagine teaching someone math: you start with 2+2,
not calculus. Curriculum learning applies this principle to machine learning:

- **Epoch 1**: Mostly easy samples (simple patterns, clear examples)
- **Epoch 5**: Mix of easy and medium samples
- **Epoch 10**: All samples including hard ones

This often leads to faster convergence and better final performance.

Example:

## How It Works

CurriculumSampler implements the idea that models learn better when training starts
with easy examples and gradually progresses to harder ones, similar to how humans
learn complex subjects step by step.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumSampler(IEnumerable<>,Int32,CurriculumStrategy,Nullable<Int32>)` | Initializes a new instance of the CurriculumSampler class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurrentDifficultyThreshold` | Gets the current difficulty threshold based on epoch and strategy. |
| `Length` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetIndicesCore` |  |
| `SetCompetence(Double)` | Sets the model's current competence level for competence-based curriculum. |

