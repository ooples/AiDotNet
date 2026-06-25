---
title: "ShuffleModelDP<T>"
description: "Implements Shuffle Model Differential Privacy for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

Implements Shuffle Model Differential Privacy for federated learning.

## For Beginners

In standard local DP, each client adds a lot of noise to their
update before sending it (because the server sees individual updates). In shuffle model DP,
a trusted shuffler randomly permutes the updates before the server sees them. Because the
server can't link updates to specific clients, each client needs to add much less noise —
achieving central-DP-level accuracy with local-DP trust assumptions.

## How It Works

Privacy amplification by shuffling:

Protocol:

Reference: Balle, B., Bell, J., & Gascon, A. (2019). "The Privacy Blanket of the
Shuffle Model." Crypto 2019.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ShuffleModelDP(Double,Int32)` | Creates a new Shuffle Model DP mechanism. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LocalEpsilon` | Gets the local epsilon value. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateShuffled(List<Dictionary<String,[]>>)` | Aggregates shuffled, anonymized updates by simple averaging. |
| `ApplyLocalDPAndShuffle(Dictionary<Int32,Dictionary<String,[]>>,Double)` | Applies local DP noise to each client's update and then shuffles the collection. |
| `ApplyPrivacy(Dictionary<String,[]>,Double,Double)` |  |
| `ComputeEffectiveEpsilon(Int32,Double)` | Computes the effective central epsilon after shuffling n clients using the tight bound from Balle, Bell, and Gascon (2019). |
| `ComputeMinClientsNeeded(Double,Double)` | Computes the minimum number of clients needed to achieve a target central epsilon. |
| `GetMechanismName` |  |
| `GetPrivacyBudgetConsumed` |  |
| `ShuffleAndAnonymize(Dictionary<Int32,Dictionary<String,[]>>)` | Shuffles a collection of client updates, removing the association between client IDs and their updates. |

