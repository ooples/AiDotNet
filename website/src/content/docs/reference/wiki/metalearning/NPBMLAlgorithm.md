---
title: "NPBMLAlgorithm<T, TInput, TOutput>"
description: "Implementation of NPBML (Neural Process-Based Meta-Learning)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of NPBML (Neural Process-Based Meta-Learning).

## For Beginners

NPBML is a probabilistic meta-learner that knows when it's uncertain:

**Standard meta-learners:**
Given a support set, produce ONE adapted model that gives ONE prediction per query.
You don't know how confident the model is.

**NPBML's probabilistic approach:**

1. Encode support examples into a DISTRIBUTION (mean + variance), not a single point
2. Sample from this distribution multiple times
3. Each sample gives a slightly different prediction
4. If samples agree: model is confident
5. If samples disagree: model is uncertain about this task

**Why this matters:**

- Few support examples = high uncertainty (wide distribution)
- Many similar support examples = low uncertainty (narrow distribution)
- Ambiguous task = high uncertainty (can't determine the right adaptation)

**Analogy:**
It's like asking 5 experts (sampled from the distribution). If all 5 agree, you're confident.
If they give different answers, you know the task is ambiguous.

## How It Works

NPBML combines neural processes with meta-learning for probabilistic few-shot prediction.
It encodes the support set into a latent distribution and samples from it to capture
task-level uncertainty.

**Algorithm - NPBML:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NPBMLAlgorithm(NPBMLOptions<,,>)` | Initializes a new NPBML meta-learner. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `ComputeAuxLoss(TaskBatch<,,>)` | Computes the average loss over a task batch using encoder/decoder + KL divergence. |
| `ComputeKLDivergence(Double[],Double[])` | Computes the KL divergence between a diagonal Gaussian and the standard normal. |
| `DecodeLatent(Vector<>,Vector<>)` | Decodes the sampled latent combined with query features through the decoder network. |
| `EncodeAndSample(Vector<>)` | Encodes the support set into a latent distribution and samples from it using the reparameterization trick. |
| `EncodeToDistribution(Vector<>)` | Encodes support features into a latent distribution (mu, log_sigma) using the encoder network. |
| `InitializeEncoderDecoder` | Initializes encoder and decoder parameters. |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_decoderParams` | Parameters for the decoder (latent + query -> predictions). |
| `_encoderParams` | Parameters for the encoder (support set -> latent distribution). |

