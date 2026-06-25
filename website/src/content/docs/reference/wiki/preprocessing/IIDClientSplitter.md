---
title: "IIDClientSplitter<T>"
description: "IID (Independent and Identically Distributed) client splitter for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.FederatedLearning`

IID (Independent and Identically Distributed) client splitter for federated learning.

## For Beginners

In federated learning, data is distributed across multiple clients
(devices or institutions). IID partitioning means each client gets a random sample
of the overall data, so all clients have similar data distributions.

## How It Works

**How It Works:**

1. Shuffle all data randomly
2. Divide equally (or according to specified ratios) among clients
3. Each client's data is statistically similar to the global distribution

**When to Use:**

- Simulating ideal federated learning scenarios
- Baseline experiments before testing non-IID
- When clients should have representative samples

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `IIDClientSplitter(Int32,Double[],Double,Boolean,Int32)` | Creates a new IID client splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |
| `NumSplits` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSplits(Matrix<>,Vector<>)` |  |
| `Split(Matrix<>,Vector<>)` |  |

