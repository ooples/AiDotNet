---
title: "QueryByCommitteeStrategy<T, TInput, TOutput>"
description: "Query By Committee (QBC) strategy for active learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActiveLearning.Strategies.Committee`

Query By Committee (QBC) strategy for active learning.

## For Beginners

QBC uses a "committee" of diverse models to identify samples
where the models disagree most. High disagreement indicates uncertainty about the true label.

## How It Works

**How QBC Works:**

**Disagreement Measures:**

**When to Use:**

**Reference:** Seung et al. "Query by Committee" (1992)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `QueryByCommitteeStrategy(IEnumerable<IFullModel<,,>>,CommitteeDisagreementMeasure,Int32,ActiveLearnerConfig<>)` | Initializes a new QBC strategy with a pre-built committee. |
| `QueryByCommitteeStrategy(Int32)` | Initializes a new QBC strategy with an empty committee. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Committee` |  |
| `Description` |  |
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCommitteeMember(IFullModel<,,>)` |  |
| `ComputeDisagreement()` |  |
| `ComputeScores(IFullModel<,,>,IDataset<,,>)` |  |
| `RemoveCommitteeMember(IFullModel<,,>)` |  |
| `Reset` |  |
| `SelectSamples(IFullModel<,,>,IDataset<,,>,Int32)` |  |
| `UpdateCommittee(IDataset<,,>)` |  |
| `UpdateState(Int32[],[])` |  |

