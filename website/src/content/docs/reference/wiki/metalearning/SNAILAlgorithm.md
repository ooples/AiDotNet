---
title: "SNAILAlgorithm<T, TInput, TOutput>"
description: "Implementation of SNAIL (Simple Neural Attentive Meta-Learner) for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of SNAIL (Simple Neural Attentive Meta-Learner) for few-shot learning.

## For Beginners

SNAIL treats few-shot learning as reading a story:

**How it works:**

1. Take all support examples and their labels, arrange them in a sequence
2. Append query examples (without labels) at the end
3. Use temporal convolutions to capture local patterns (nearby examples)
4. Use causal attention to capture global patterns (any previous example)
5. The model predicts labels for query examples using what it "remembers"

**Analogy:**
Imagine reading a detective novel:

- Support examples are like clues scattered through the story
- Temporal convolutions help you remember recent clues (short-term memory)
- Causal attention lets you recall any clue from the entire story (long-term memory)
- At the end (query), you must solve the mystery using all clues gathered

**Key insight:**
By treating few-shot learning as sequence modeling, SNAIL can leverage powerful
sequence architectures (temporal convolutions + attention) that have been very
successful in NLP and speech recognition.

**Architecture:**

- Temporal Convolution (TC) blocks: Dilated causal convolutions that aggregate local info
- Causal Attention blocks: Attend to any previous position in the sequence
- Stacked TC + Attention blocks form the full SNAIL architecture

## How It Works

SNAIL combines temporal convolutions with causal attention to process few-shot learning
as a sequence modeling problem. Support examples (with labels) are fed sequentially,
then query examples are fed to produce predictions. The model learns which examples to
remember and attend to.

**Algorithm - SNAIL:**

Reference: Mishra, N., Rohaninejad, M., Chen, X., & Abbeel, P. (2018).
A Simple Neural Attentive Learner. ICLR 2018.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SNAILAlgorithm(SNAILOptions<,,>)` | Initializes a new SNAIL meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` | Adapts to a new task using SNAIL's sequence processing. |
| `ApplyAttentionBlock(Vector<>,Vector<>)` | Applies a causal attention block. |
| `ApplyTCBlock(Vector<>,Vector<>)` | Applies a temporal convolution block with dilated causal convolutions. |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using SNAIL block processing. |
| `InitializeSNAILBlocks` | Initializes the temporal convolution and attention block parameters. |
| `MetaTrain(TaskBatch<,,>)` | Performs one meta-training step for SNAIL. |
| `ProcessSNAILBlocks(,)` | Processes inputs through the stacked TC + Attention blocks. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_attentionBlockParams` | Parameters for the causal attention blocks. |
| `_isTraining` | Whether the model is in training mode (affects dropout behavior). |
| `_tcBlockParams` | Parameters for the temporal convolution blocks. |

