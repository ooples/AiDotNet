---
title: "NonIIDClientSplitter<T>"
description: "Non-IID (non-Independent and Identically Distributed) client splitter for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.FederatedLearning`

Non-IID (non-Independent and Identically Distributed) client splitter for federated learning.

## For Beginners

In real-world federated learning, data on different clients
is often NOT identically distributed. For example, users' photos on different phones
reflect their individual preferences. This splitter simulates such heterogeneous distributions.

## How It Works

**Heterogeneity Types:**

- Label skew: Each client has only some classes
- Quantity skew: Clients have different amounts of data
- Feature skew: Clients have different feature distributions

**When to Use:**

- Realistic federated learning experiments
- Testing robustness to heterogeneous data
- Simulating domain adaptation scenarios

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NonIIDClientSplitter(Int32,Int32,Double,Boolean,Boolean,Int32)` | Creates a new Non-IID client splitter. |

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

