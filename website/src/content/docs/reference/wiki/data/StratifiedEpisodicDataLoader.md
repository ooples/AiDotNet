---
title: "StratifiedEpisodicDataLoader<T, TInput, TOutput>"
description: "Provides stratified episodic task sampling that maintains dataset class proportions across tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Provides stratified episodic task sampling that maintains dataset class proportions across tasks.

## For Beginners

Real-world datasets often have imbalanced class distributions:

- Medical datasets might have 90% healthy cases, 10% disease cases
- E-commerce might have 80% browsing, 15% cart additions, 5% purchases
- Image datasets might have common objects (cars, trees) and rare ones (exotic animals)

Standard random sampling treats all classes equally, which doesn't reflect reality.
Stratified sampling maintains these natural proportions:

- Common classes appear in more tasks
- Rare classes appear in fewer tasks
- The model learns to handle the real-world distribution

**When to use this:**

- When your dataset has natural class imbalance that you want to preserve
- When training for real-world deployment where class frequencies matter
- When you want meta-learning to reflect actual data distributions

**When NOT to use this:**

- When you want equal exposure to all classes (use BalancedEpisodicDataLoader)
- When evaluating few-shot learning fairly across all classes
- When class frequencies in deployment differ from training data

## How It Works

The StratifiedEpisodicDataLoader extends the standard episodic loader by sampling classes
proportionally to their representation in the dataset. If a class represents 30% of the dataset,
it will appear in approximately 30% of tasks over many episodes, preserving the natural
class distribution.

**Thread Safety:** This class is not thread-safe due to internal Random state.
Create separate instances for concurrent task generation.

**Performance:** Similar to standard EpisodicDataLoader with O(nWay × (kShot + queryShots))
complexity. Slightly slower due to proportional weight calculation during initialization.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StratifiedEpisodicDataLoader(Matrix<>,Vector<>,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the StratifiedEpisodicDataLoader for proportional N-way K-shot task sampling. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNextTaskCore` | Core implementation of stratified N-way K-shot task sampling with proportional class selection. |
| `WeightedSampleClasses(Int32)` | Performs weighted random sampling of classes based on their dataset proportions. |

