---
title: "CurriculumEpisodicDataLoader<T, TInput, TOutput>"
description: "Provides curriculum-based episodic task sampling that progressively increases task difficulty during training."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Provides curriculum-based episodic task sampling that progressively increases task difficulty during training.

## For Beginners

Curriculum learning is inspired by how humans learn:

- You don't start learning math with calculus - you start with counting, then addition, etc.
- You don't learn a language by reading novels - you start with basic vocabulary and grammar
- Complex skills are built on simpler foundations

This loader applies the same principle to meta-learning:

- **Easy tasks:** 2-way 10-shot (2 classes, 10 examples each) - lots of examples, few classes
- **Medium tasks:** 4-way 5-shot (4 classes, 5 examples each) - balanced difficulty
- **Hard tasks:** 5-way 1-shot (5 classes, 1 example each) - few examples, many classes

**How difficulty progression works:**

- Progress is tracked from 0.0 (start) to 1.0 (end)
- At progress 0.0: Easy tasks (2-way, lots of shots)
- At progress 0.5: Medium tasks (halfway to target)
- At progress 1.0: Target difficulty (full N-way K-shot)

**When to use this:**

- When training struggles to converge on hard tasks from the start
- When you want to improve sample efficiency and training stability
- When implementing human-inspired learning strategies
- Research has shown curriculum learning often leads to better final performance

**Note:** You must manually update the progress as training proceeds (e.g., based on
episode number, training loss, or validation accuracy).

## How It Works

The CurriculumEpisodicDataLoader implements curriculum learning for meta-learning by starting
with easier tasks and progressively increasing difficulty. Easier tasks have fewer classes (lower N-way)
and more examples per class (higher K-shot), while harder tasks approach the target N-way K-shot configuration.
This gradual progression helps models learn more effectively by building on simpler concepts first.

**Thread Safety:** This class is not thread-safe due to internal state.
Create separate instances for concurrent task generation.

**Performance:** Same complexity as standard EpisodicDataLoader, with minor overhead
for calculating current difficulty parameters based on progress.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CurriculumEpisodicDataLoader(Matrix<>,Vector<>,Int32,Int32,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the CurriculumEpisodicDataLoader for progressive N-way K-shot task sampling. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CurriculumProgress` | Gets the current curriculum progress (0.0 = easiest, 1.0 = target difficulty). |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateCurrentKShot` | Calculates the current K-shot based on curriculum progress. |
| `CalculateCurrentNWay` | Calculates the current N-way based on curriculum progress. |
| `GetNextTaskCore` | Core implementation of curriculum-based N-way K-shot task sampling with progressive difficulty. |
| `SetProgress(Double)` | Sets the curriculum progress to control task difficulty. |

