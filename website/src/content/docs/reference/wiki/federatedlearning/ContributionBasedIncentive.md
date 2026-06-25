---
title: "ContributionBasedIncentive<T>"
description: "Incentive mechanism that rewards clients proportional to their evaluated contribution."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Incentive mechanism that rewards clients proportional to their evaluated contribution.

## For Beginners

Clients need motivation to participate in federated learning.
This mechanism distributes a reward budget proportionally: clients who contribute more
valuable data get a bigger share. It also tracks trust over time — consistently helpful
clients earn higher trust scores, while erratic or low-quality contributors are flagged.

## How It Works

**How rewards work:**

**Trust scoring:** Trust is computed from contribution history using exponential
moving average with consistency bonus. Clients with stable, positive contributions earn
higher trust than those with sporadic or negative contributions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ContributionBasedIncentive(ContributionEvaluationOptions)` | Initializes a new instance of `ContributionBasedIncentive`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MechanismName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeRewards(Dictionary<Int32,Double>,Double)` |  |
| `ComputeTrustScores(Dictionary<Int32,List<Double>>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinRewardFraction` | Minimum reward fraction allocated to each participating client (prevents zero rewards). |
| `TrustDecay` | Decay factor for exponential moving average in trust computation. |

