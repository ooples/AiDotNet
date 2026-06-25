---
title: "MetaCollaborativeAlgorithm<T, TInput, TOutput>"
description: "Implementation of Meta-Collaborative Learning for cross-domain few-shot learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.MetaLearning.Algorithms`

Implementation of Meta-Collaborative Learning for cross-domain few-shot learning.

## How It Works

Meta-Collaborative Learning maintains a set of domain-specific gradient momentum buffers
and uses gradient alignment (cosine similarity) between tasks to modulate cross-task
knowledge transfer. Tasks with well-aligned gradients reinforce each other's updates;
conflicting gradients are dampened via a PCGrad-inspired projection.

**Algorithm:**

## Properties

| Property | Summary |
|:-----|:--------|
| `AlgorithmType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Adapt(IMetaLearningTask<,,>)` |  |
| `AssignToDomains(List<Vector<>>)` | Assigns each task to the domain slot with highest gradient cosine similarity. |
| `ComputeCollaborativeGradient(Vector<>,Int32)` | Computes collaborative gradient: adds aligned domain signals, projects out conflicting ones. |
| `MetaTrain(TaskBatch<,,>)` |  |
| `UpdateDomainBuffers(List<Vector<>>,Int32[])` | Updates domain buffers with EMA of assigned task gradients. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_domainBuffers` | Domain-specific gradient momentum buffers. |

