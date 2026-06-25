---
title: "PTMAPAlgorithm<T, TInput, TOutput>"
description: "Implementation of PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of PT+MAP (Power Transform + Maximum A Posteriori) (Hu et al., ICLR 2021).

## For Beginners

PT+MAP is elegantly simple:

**Step 1 - Power Transform:**
Raw features from neural networks often have skewed distributions.
The power transform x_new = sign(x) * |x|^beta makes them more bell-shaped (Gaussian).
With beta=0.5, this is essentially a square root transform.

**Step 2 - Center and Normalize:**
After the transform, center features (subtract mean) and L2 normalize.

**Step 3 - MAP Estimation (transductive):**
Given the Gaussian assumption, compute the optimal (MAP) class assignments
for ALL query examples simultaneously. This iterates between:

- Assign queries to nearest class (E-step)
- Update class means using assigned queries (M-step)

**Why it works so well:**
The power transform fixes the main problem: features aren't Gaussian.
Once they ARE Gaussian, the simple MAP classifier is provably optimal.
Sometimes the simplest math wins.

## How It Works

PT+MAP applies a power transform to normalize feature distributions, then uses
MAP estimation for transductive few-shot classification. The power transform makes
features more Gaussian, enabling a simple Bayesian classifier to work well.

**Algorithm - PT+MAP:**

Reference: Hu, Y., Gripon, V., & Pateux, S. (2021).
Leveraging the Feature Distribution in Transfer-based Few-Shot Learning. ICLR 2021.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PTMAPAlgorithm(PTMAPOptions<,,>)` | Initializes a new PT+MAP meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ApplyPowerTransform(Vector<>)` | Applies the power transform to a feature vector. |
| `CenterAndNormalize(Vector<>)` | Centers and L2-normalizes a feature vector. |
| `MAPEstimation(Vector<>,Vector<>)` | Performs MAP estimation with iterative refinement on query soft assignments. |
| `MetaTrain(TaskBatch<,,>)` |  |

