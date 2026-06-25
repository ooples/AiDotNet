---
title: "Samplers"
description: "Static factory class for creating data samplers with beginner-friendly methods."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Data.Sampling`

Static factory class for creating data samplers with beginner-friendly methods.

## For Beginners

Sampling strategies determine how you pick data from your dataset.
Different strategies can help with:

- Class imbalance (use weighted sampling)
- Curriculum learning (start with easy examples, progress to hard)
- Active learning (focus on uncertain examples)

**Common Patterns:**
```cs
// Random sampling (default, good for most cases)
var sampler = Samplers.Random(dataSize);

// Balanced sampling for imbalanced classes
var sampler = Samplers.Balanced(labels, numClasses);

// Curriculum learning (easy to hard)
var sampler = Samplers.Curriculum(difficulties, totalEpochs);
```

## How It Works

Samplers provides factory methods for creating various sampling strategies used
during training. These samplers control which data points are selected and in what order.

## Methods

| Method | Summary |
|:-----|:--------|
| `ActiveLearning(Int32,ActiveLearningStrategy,Double,Nullable<Int32>)` | Creates an active learning sampler that prioritizes uncertain samples. |
| `ActiveLearning(Int32,ActiveLearningStrategy,Double,Nullable<Int32>)` | Creates an active learning sampler that prioritizes uncertain samples. |
| `Balanced(IReadOnlyList<Int32>,Int32,Nullable<Int32>)` | Creates a balanced sampler that oversamples minority classes. |
| `Curriculum(IEnumerable<>,Int32,CurriculumStrategy,Nullable<Int32>)` | Creates a curriculum learning sampler that starts with easy samples. |
| `Curriculum(IEnumerable<Double>,Int32,CurriculumStrategy,Nullable<Int32>)` | Creates a curriculum learning sampler that starts with easy samples. |
| `Importance(Int32,Double,Boolean,Nullable<Int32>)` | Creates an importance sampler that prioritizes high-loss samples. |
| `Importance(Int32,Double,Boolean,Nullable<Int32>)` | Creates an importance sampler that prioritizes high-loss samples. |
| `Random(Int32,Nullable<Int32>)` | Creates a random sampler that shuffles data each epoch. |
| `SelfPaced(Int32,,,Int32,Nullable<Int32>)` | Creates a self-paced learning sampler that adapts based on model performance. |
| `SelfPaced(Int32,Double,Double,Int32,Nullable<Int32>)` | Creates a self-paced learning sampler with default parameters. |
| `Sequential(Int32)` | Creates a sequential sampler that iterates through data in order. |
| `Stratified(IReadOnlyList<Int32>,Int32,Nullable<Int32>)` | Creates a stratified sampler that maintains class proportions in each batch. |
| `Subset(IEnumerable<Int32>,Boolean,Nullable<Int32>)` | Creates a subset sampler that samples from specific indices. |
| `Weighted(IEnumerable<>,Int32,Boolean,Nullable<Int32>)` | Creates a weighted sampler that samples based on per-sample weights. |

