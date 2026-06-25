---
title: "FederatedRLHF<T>"
description: "Configuration and orchestration for Federated RLHF (Reinforcement Learning from Human Feedback)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Alignment`

Configuration and orchestration for Federated RLHF (Reinforcement Learning from Human Feedback).

## For Beginners

RLHF is how modern LLMs learn to be helpful, harmless, and honest.
Normally, human feedback data is centralized. Federated RLHF keeps the feedback private at each
organization: each client trains a local reward model on their preference data, and the server
aggregates these reward models. The policy (LLM) is then fine-tuned using the aggregated reward
signal via PPO or similar RL algorithms.

## How It Works

Pipeline:

Reference: Federated RLHF for Privacy-Preserving LLM Alignment (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedRLHF(FederatedRLHFOptions)` | Creates a new Federated RLHF orchestrator. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the RLHF configuration options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateRewardModels(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates reward model parameters from multiple clients. |
| `ComputeGAE(Double[],Double[],Double,Double)` | Computes GAE (Generalized Advantage Estimation) for PPO training. |
| `ComputeKLPenalty(Double[],Double[])` | Computes the KL penalty for PPO-style policy updates. |
| `ComputePPOLoss(Double[],Double[],Double[])` | Computes the PPO clipped surrogate loss for a batch of tokens. |
| `ComputeRewardsWithKLPenalty(Double[],Double[],Double[])` | Computes per-token rewards by combining reward model scores with KL penalty. |

