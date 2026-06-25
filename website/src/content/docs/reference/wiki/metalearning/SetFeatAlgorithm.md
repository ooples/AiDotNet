---
title: "SetFeatAlgorithm<T, TInput, TOutput>"
description: "Implementation of SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of SetFeat (set-feature based few-shot learning) (Afrasiyabi et al., CVPR 2022).

## For Beginners

SetFeat treats each class as a SET, not just a single point:

**The problem with prototypes:**
ProtoNets computes the MEAN of support examples. This throws away information about
HOW the class varies. Two classes might have the same mean but very different spreads.

**How SetFeat fixes this:**

1. Extract features for each class's support examples
2. Feed ALL examples (as a set) into a set encoder
3. The set encoder captures rich information: mean, variance, relationships
4. Optional cross-attention lets classes "see" each other for context
5. The resulting set-features are used for classification

**Example:**
If you have 5 examples of cats (tabby, persian, siamese, calico, sphinx):

- ProtoNets: Average them into one "generic cat" point
- SetFeat: Encodes that cats come in different fur patterns and body types

This extra information helps distinguish cats from similar classes like small dogs.

## How It Works

SetFeat learns set-level features by processing each class's support examples as a set
rather than individual instances. A set encoder with optional cross-attention computes
class representations that capture intra-class variation.

**Algorithm - SetFeat:**

Reference: Afrasiyabi, A., Larochelle, H., Lalonde, J.F., & Gagne, C. (2022).
Matching Feature Sets for Few-Shot Image Classification. CVPR 2022.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SetFeatAlgorithm(SetFeatOptions<,,>)` | Initializes a new SetFeat meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ApplyCrossAttention(Vector<>)` | Applies cross-attention between class representations, allowing each class to adjust based on all other classes. |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using set encoding + cross-attention. |
| `EncodeSet(Vector<>)` | Encodes a set of features using the set encoder (attention pooling). |
| `InitializeSetEncoder` | Initializes set encoder and cross-attention parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_crossAttentionParams` | Parameters for the cross-attention module. |
| `_setEncoderParams` | Parameters for the set encoder. |

