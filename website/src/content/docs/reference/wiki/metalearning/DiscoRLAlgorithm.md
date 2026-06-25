---
title: "DiscoRLAlgorithm<T, TInput, TOutput>"
description: "Implementation of DiscoRL: Discovery-based meta-RL with reusable skill discovery."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of DiscoRL: Discovery-based meta-RL with reusable skill discovery.

## How It Works

DiscoRL discovers reusable "skills" as low-rank directions in parameter space. Each
skill is a rank-R basis spanning a subspace of the full parameter space. A gating
network selects which skills to activate based on the initial task gradient signal.
During adaptation, only the skill coefficients are updated, enabling efficient reuse.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `MetaTrain(TaskBatch<,,>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_gatingParams` | Gating network parameters: compressedDim * numSkills. |
| `_skillBasis` | Skill basis vectors: numSkills * skillRank * compressedDim. |

