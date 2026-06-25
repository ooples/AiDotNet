---
title: "IIncentiveMechanism<T>"
description: "Computes incentive rewards for federated learning participants based on their contributions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Fairness`

Computes incentive rewards for federated learning participants based on their contributions.

## For Beginners

Clients need motivation to participate in federated learning.
Without incentives, rational clients may free-ride (benefit from the global model without
contributing their own data). An incentive mechanism assigns rewards proportional to each
client's contribution, encouraging high-quality participation.

## How It Works

**Example:** In a data marketplace, a hospital contributing rare disease data
(high marginal value) earns more than one contributing common cold data (low marginal value).

## Properties

| Property | Summary |
|:-----|:--------|
| `MechanismName` | Gets the name of this incentive mechanism. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeRewards(Dictionary<Int32,Double>,Double)` | Computes incentive rewards for each client based on their contribution scores. |
| `ComputeTrustScores(Dictionary<Int32,List<Double>>)` | Computes trust scores for clients based on their participation history. |

