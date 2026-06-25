---
title: "FedFairOptimizer<T>"
description: "Implements FedFair — multi-objective optimization balancing accuracy, fairness, and efficiency."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Fairness`

Implements FedFair — multi-objective optimization balancing accuracy, fairness, and efficiency.

## For Beginners

Real FL deployments must balance multiple goals: high accuracy
(model quality), fairness (no client left behind), and efficiency (fast convergence, low
communication). FedFair treats these as a multi-objective optimization problem and finds
Pareto-optimal aggregation weights that don't sacrifice one goal unnecessarily for another.
The user specifies preference weights for each objective.

## How It Works

Objectives:

Scalarization: w = alpha_acc * w_acc + alpha_fair * w_fair + alpha_eff * w_eff

Reference: FedFair: Multi-Objective Federated Learning (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FedFairOptimizer(Double,Double,Double)` | Creates a new FedFair optimizer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AccuracyWeight` | Gets the accuracy preference weight. |
| `EfficiencyWeight` | Gets the efficiency preference weight. |
| `FairnessWeight` | Gets the fairness preference weight. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeWeights(Dictionary<Int32,Double>,Dictionary<Int32,Int32>,Dictionary<Int32,Double>)` | Computes multi-objective aggregation weights. |

