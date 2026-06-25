---
title: "SNAILOptions<T, TInput, TOutput>"
description: "Configuration options for SNAIL (Simple Neural Attentive Meta-Learner) algorithm."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.MetaLearning.Options`

Configuration options for SNAIL (Simple Neural Attentive Meta-Learner) algorithm.

## For Beginners

SNAIL treats few-shot learning as a sequence problem:

**The Idea:**
Feed all support examples (with their labels) one by one into the model,
then feed query examples (without labels). The model learns to:

1. Remember important examples (using temporal convolutions)
2. Focus on relevant ones (using attention)
3. Predict labels for query examples

**Analogy:**
Imagine reading a detective story:

- Support examples are like clues presented throughout the story
- Temporal convolutions help you remember clues from different points in time
- Attention helps you focus on the most relevant clues when solving the mystery
- The query is the mystery to solve, using all the clues you've gathered

**Architecture:**

- Temporal Convolutions (TC): Capture short-range dependencies in the example sequence
- Causal Attention: Capture long-range dependencies by attending to any previous example
- Together: TC for local patterns + Attention for global patterns = powerful meta-learner

## How It Works

SNAIL combines temporal convolutions with causal attention to perform sequence-to-sequence
meta-learning. It processes support examples as a sequence and uses attention to selectively
recall relevant examples when classifying query examples.

Reference: Mishra, N., Rohaninejad, M., Chen, X., & Abbeel, P. (2018).
A Simple Neural Attentive Learner. ICLR 2018.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SNAILOptions(IFullModel<,,>)` | Initializes a new instance of SNAILOptions. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdaptationSteps` |  |
| `AttentionKeyDim` | Gets or sets the key dimension for attention. |
| `AttentionValueDim` | Gets or sets the value dimension for attention. |
| `CheckpointFrequency` |  |
| `DataLoader` | Gets or sets the episodic data loader. |
| `DropoutRate` | Gets or sets the dropout rate for temporal convolutions and attention. |
| `EnableCheckpointing` |  |
| `EvaluationFrequency` |  |
| `EvaluationTasks` |  |
| `GradientClipThreshold` |  |
| `InnerLearningRate` |  |
| `InnerOptimizer` | Gets or sets the inner loop optimizer. |
| `LossFunction` | Gets or sets the loss function. |
| `MaxSequenceLength` | Gets or sets the maximum sequence length the model can handle. |
| `MetaBatchSize` |  |
| `MetaModel` | Gets or sets the base feature extractor model. |
| `MetaOptimizer` | Gets or sets the outer loop optimizer. |
| `NumAttentionHeads` | Gets or sets the number of attention heads for the causal attention blocks. |
| `NumBlocks` | Gets or sets the number of TC+Attention blocks to stack. |
| `NumMetaIterations` |  |
| `NumTCFilters` | Gets or sets the number of temporal convolution filters per block. |
| `OuterLearningRate` |  |
| `RandomSeed` | Gets or sets the random seed. |
| `UseFirstOrder` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` |  |
| `IsValid` |  |

