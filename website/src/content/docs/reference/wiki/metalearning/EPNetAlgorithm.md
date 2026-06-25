---
title: "EPNetAlgorithm<T, TInput, TOutput>"
description: "Implementation of EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of EPNet (Embedding Propagation Network) (Rodriguez et al., CVPR 2020).

## For Beginners

EPNet makes features better by sharing information:

**The insight:**
Features extracted by a neural network are good but noisy. If two examples are
similar, they should have similar features. EPNet enforces this by propagating
features through a similarity graph.

**How it works:**

1. Extract features for all examples (support + query)
2. Build a k-nearest-neighbor graph based on feature similarity
3. Propagate features through the graph (like heat diffusion)
- Each node averages its neighbors' features (weighted by similarity)
- Repeat for several iterations
4. The propagated features are smoother and more consistent
5. Classify using the refined features

**Why propagation helps:**

- Noisy features get smoothed out (noise reduction)
- Cluster structure becomes clearer (better separation)
- Query examples near support clusters get pulled toward them
- Works transductively: ALL queries benefit from each other

## How It Works

EPNet refines embeddings through label propagation on a nearest-neighbor graph.
By propagating feature information between similar examples, embeddings become
smoother and more discriminative for few-shot classification.

**Algorithm - EPNet:**

Reference: Rodriguez, P., Laradji, I., Drouin, A., & Lacoste, A. (2020).
Embedding Propagation: Smoother Manifold for Few-Shot Classification. CVPR 2020.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `EPNetAlgorithm(EPNetOptions<,,>)` | Initializes a new EPNet meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |
| `PropagateEmbeddings(Vector<>)` | Performs embedding propagation on a feature vector using a proper kNN similarity graph. |

