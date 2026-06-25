---
title: "DirichletSplitter<T>"
description: "Dirichlet distribution-based splitter for federated learning with controlled heterogeneity."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.FederatedLearning`

Dirichlet distribution-based splitter for federated learning with controlled heterogeneity.

## For Beginners

The Dirichlet distribution is used to control how "unequal" the
class distributions are across clients. A concentration parameter (alpha) controls the
heterogeneity: smaller alpha means more heterogeneous (clients have very different distributions),
larger alpha means more homogeneous (clients have similar distributions).

## How It Works

**Alpha Values:**

- alpha = 0.1: Extreme heterogeneity (clients may have only 1-2 classes)
- alpha = 1.0: Moderate heterogeneity
- alpha = 10.0: Nearly IID (all clients have similar class distributions)
- alpha = 100.0: Practically IID

**When to Use:**

- Systematic federated learning experiments
- Studying effect of heterogeneity levels
- Reproducing research paper setups

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DirichletSplitter(Int32,Double,Double,Int32,Int32)` | Creates a new Dirichlet distribution splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |
| `RequiresLabels` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

