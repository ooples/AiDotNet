---
title: "FreqPromptAlgorithm<T, TInput, TOutput>"
description: "Implementation of FreqPrompt: Frequency-domain prompt tuning for few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of FreqPrompt: Frequency-domain prompt tuning for few-shot learning.

## How It Works

FreqPrompt meta-learns a set of frequency basis vectors and their coefficients that
act as additive "prompts" in the parameter space. During adaptation, only the prompt
coefficients are updated (the backbone is frozen), which enables efficient adaptation
with very few parameters. Low-frequency prompts capture coarse domain shifts while
high-frequency prompts handle fine-grained adjustments.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeEffectiveParams(Vector<>,Double[])` | Computes effective parameters: θ = θ_base + Σ_k c_k * B_k. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_freqWeights` | Per-frequency regularization weights (higher for high-freq). |
| `_promptBasis` | Prompt basis vectors: flat array of K * paramDim values. |
| `_promptCoeffsInit` | Meta-learned initial prompt coefficients: length K. |

