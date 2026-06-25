---
title: "BalancedEpisodicDataLoader<T, TInput, TOutput>"
description: "Provides balanced episodic task sampling that ensures equal class representation across multiple tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Loaders`

Provides balanced episodic task sampling that ensures equal class representation across multiple tasks.

## For Beginners

Standard random sampling might pick some classes more often than others
by chance. This can cause problems in meta-learning:

- The model might learn some classes better than others
- Training could be biased toward frequently-sampled classes
- Evaluation metrics might be skewed

The balanced loader solves this by:

- Tracking how many times each class has been selected
- Preferring classes that haven't been used as much
- Ensuring fair representation across all classes over time

**When to use this:**

- Long meta-training runs where balanced class exposure matters
- When your dataset has many classes and you want uniform coverage
- When evaluating meta-learning algorithms fairly across all classes

**Trade-off:** Less random than uniform sampling, but more balanced. Good for training,
but you might want standard EpisodicDataLoader for final evaluation to match real-world randomness.

## How It Works

The BalancedEpisodicDataLoader extends the standard episodic loader by tracking class usage
across multiple tasks and preferentially sampling under-represented classes. This ensures
that over many episodes, all classes appear roughly the same number of times, preventing
bias toward frequently-sampled classes.

**Thread Safety:** This class is not thread-safe due to internal state tracking.
Create separate instances for concurrent task generation.

**Performance:** Slightly slower than standard EpisodicDataLoader due to usage tracking
and weighted sampling, but still O(nWay × (kShot + queryShots)) for task creation.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BalancedEpisodicDataLoader(Matrix<>,Vector<>,Int32,Int32,Int32,Nullable<Int32>)` | Initializes a new instance of the BalancedEpisodicDataLoader for balanced N-way K-shot task sampling. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNextTaskCore` | Core implementation of balanced N-way K-shot task sampling with weighted class selection. |
| `WeightedSample(Int32[],Dictionary<Int32,Double>,Int32)` | Performs weighted random sampling without replacement. |

