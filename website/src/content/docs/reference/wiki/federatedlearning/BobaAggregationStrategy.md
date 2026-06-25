---
title: "BobaAggregationStrategy<T>"
description: "Implements BOBA (Bayesian Optimal Byzantine-robust Aggregation) strategy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Aggregators`

Implements BOBA (Bayesian Optimal Byzantine-robust Aggregation) strategy.

## For Beginners

Most Byzantine defenses use fixed rules (e.g., remove outliers).
BOBA takes a probabilistic approach using a two-class mixture model — it models client updates
as coming from either an "honest" distribution (tight cluster) or a "Byzantine" distribution
(diffuse/adversarial). It uses Expectation-Maximization (EM) to estimate which class each
client belongs to, then aggregates using only the honest-class posterior probabilities.

## How It Works

Two-class mixture model:

EM algorithm per round:

Cross-round belief propagation: posteriors from round t become priors for round t+1,
so persistent attackers accumulate low trust over multiple rounds.

Reference: BOBA: Bayesian Optimal Byzantine-robust Aggregation (2024).
https://arxiv.org/abs/2312.09672

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `BobaAggregationStrategy(Double,Int32,Double,Double)` | Initializes a new instance of the `BobaAggregationStrategy` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EmConvergenceTol` | Gets the EM convergence tolerance. |
| `EmIterations` | Gets the maximum number of EM iterations per round. |
| `PriorHonest` | Gets the prior probability of a client being honest. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` |  |
| `GetStrategyName` |  |
| `RunEM(Double[][],List<Int32>,Int32,Int32)` | Runs the EM algorithm to estimate honest-class responsibilities for each client. |

