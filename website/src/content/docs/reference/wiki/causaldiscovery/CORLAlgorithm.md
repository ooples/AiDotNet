---
title: "CORLAlgorithm<T>"
description: "CORL — Causal Ordering via Reinforcement Learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalDiscovery.ContinuousOptimization`

CORL — Causal Ordering via Reinforcement Learning.

## For Beginners

Instead of directly optimizing a weight matrix (like NOTEARS),
CORL learns the ORDER in which variables cause each other. It uses a technique from
AI game-playing (reinforcement learning) where the algorithm tries different orderings
and gets "rewarded" for finding ones that explain the data well. Once you know the
order (e.g., X causes Y which causes Z), finding the exact relationships is easy.

## How It Works

CORL learns a causal ordering of variables using a policy gradient approach inspired
by reinforcement learning. The algorithm maintains a scoring function for each position
in the ordering and uses policy gradient updates to improve the ordering based on
the BIC score of the resulting DAG. Once the ordering is determined, edge weights
are learned via OLS regression.

**Algorithm:**

- Initialize a score matrix S[i,j] = probability of variable i at position j
- Sample an ordering from the score matrix (using softmax)
- Given the ordering, compute the optimal DAG via greedy parent selection with BIC
- Use the BIC score as a reward signal
- Update S using REINFORCE-style policy gradient: S += lr * (R - baseline) * grad_log_pi
- Repeat for multiple episodes, tracking the best ordering found
- Return the DAG from the best ordering

Reference: Wang et al. (2021), "Ordering-Based Causal Discovery with Reinforcement
Learning", IJCAI.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CORLAlgorithm(CausalDiscoveryOptions)` | Initializes CORL with optional configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |
| `SupportsNonlinear` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildDAGFromOrdering(Matrix<>,Int32[],Int32,Int32,Matrix<>)` | Builds the final DAG from the best ordering using the same parent selection as scoring. |
| `ComputeBICForParents(Matrix<>,Int32,List<Int32>,Int32)` | Computes BIC score for a target with given parents using multivariate OLS. |
| `ComputeOrderingReward(Matrix<>,Int32[],Int32,Int32,Matrix<>)` | Computes the total BIC reward for a given ordering. |
| `ComputeRSS(Matrix<>,Int32,List<Int32>,Int32)` | Computes residual sum of squares for target given parents via multivariate OLS. |
| `DiscoverStructureCore(Matrix<>)` |  |
| `SampleOrdering(Matrix<>,Int32,Random)` | Samples an ordering using softmax over position scores. |
| `SelectParentsBIC(Matrix<>,Int32,Int32[],Int32,Int32)` | Selects parents for a target from predecessors in the ordering using BIC-based forward selection. |

