---
title: "OpenMAMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of Open-MAML (open-set MAML with out-of-distribution detection)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Open-MAML (open-set MAML with out-of-distribution detection).

## For Beginners

Standard few-shot learning assumes all query examples belong
to one of the support classes. But in the real world, you might encounter unknown examples.

**The problem:**
You're given 5 classes of animals to learn. Then a query comes in - it might be one of
those 5 animals, or it might be something completely different (like a car). Standard
few-shot methods will force a classification into one of the 5 classes.

**How Open-MAML fixes this:**

1. Train with MAML as usual for inner-loop adaptation
2. During meta-training, some tasks include OOD examples (open-set fraction)
3. Learn a confidence threshold: if max prediction probability < threshold, reject
4. The model learns to be uncertain about OOD examples while confident about in-distribution ones

**Key difference from MAML:**
Open-MAML adds OOD detection training, so the adapted model can say "I don't know"
instead of being forced to choose a known class.

## How It Works

Open-MAML extends MAML to handle open-set scenarios where query examples may belong
to classes not present in the support set. It adds a confidence-based rejection
mechanism to detect out-of-distribution (OOD) examples.

**Algorithm - Open-MAML:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `OpenMAMLAlgorithm(OpenMAMLOptions<,,>)` | Initializes a new Open-MAML meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

