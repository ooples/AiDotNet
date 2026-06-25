---
title: "UniformEpisodicDataLoader<T, TInput, TOutput>"
description: "Provides uniform random episodic task sampling for N-way K-shot meta-learning scenarios."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Provides uniform random episodic task sampling for N-way K-shot meta-learning scenarios.

## For Beginners

Meta-learning is training an AI to be a "fast learner." Instead of
training a model once on lots of data, we train it on many small tasks, each with very few examples.

This loader helps create those small tasks:

- **N-way:** How many different categories each task should have (e.g., 5-way = 5 classes)
- **K-shot:** How many examples per category to use for learning (e.g., 3-shot = 3 examples/class)
- **Query shots:** How many examples per category to use for testing (e.g., 10 queries/class)

Example: 5-way 3-shot with 10 queries

- Support set: 5 classes × 3 examples = 15 total examples to learn from
- Query set: 5 classes × 10 examples = 50 total examples to test on

Why this matters:

- Mimics real-world scenarios where you have limited labeled data
- Teaches models to generalize from very few examples
- Enables rapid adaptation to new tasks

The same underlying dataset is resampled many times to create different tasks, each
presenting a unique few-shot learning challenge.

## How It Works

The UniformEpisodicDataLoader transforms a standard supervised learning dataset (features + labels) into
a stream of meta-learning tasks using uniform random sampling. Each task contains a support set (for quick adaptation) and a
query set (for evaluation), enabling algorithms like MAML, Reptile, and SEAL to learn how to
learn from limited examples.

**Thread Safety:** This class is not thread-safe due to internal Random state.
Create separate instances for concurrent task generation.

**Performance:** Task creation is O(nWay × (kShot + queryShots)) for sampling and
tensor construction. Preprocessing is O(n) where n is dataset size.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniformEpisodicDataLoader(Matrix<>,Vector<>,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the UniformEpisodicDataLoader for N-way K-shot task sampling with industry-standard defaults. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNextTaskCore` | Core implementation of N-way K-shot task sampling with uniform random selection. |

