---
title: "FederatedDPO<T>"
description: "Implements Federated DPO (Direct Preference Optimization) for reward-model-free LLM alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Alignment`

Implements Federated DPO (Direct Preference Optimization) for reward-model-free LLM alignment.

## For Beginners

DPO is a simpler alternative to RLHF that skips the reward model
entirely. Instead of training a separate reward model and then using RL, DPO directly
optimizes the LLM to prefer good responses over bad ones using a binary cross-entropy loss
on preference pairs. Federated DPO lets each organization keep their preference data private
while collaboratively aligning the model.

## How It Works

DPO loss per preference pair (w, l):

where w is the preferred response, l is the dispreferred, and beta controls sharpness.

Reference: FedDPO: Federated Direct Preference Optimization (2024).
https://arxiv.org/abs/2404.18567

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FederatedDPO(FederatedDPOOptions)` | Creates a new Federated DPO instance. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Options` | Gets the DPO configuration options. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AggregateModels(Dictionary<Int32,Dictionary<String,[]>>,Dictionary<Int32,Double>)` | Aggregates DPO-trained model updates from multiple clients. |
| `ComputeDPOGradientWeights(Double[],Double[],Double[],Double[])` | Computes per-example DPO gradients (as implicit reward margins) for parameter updates. |
| `ComputeDPOLoss(Double[],Double[],Double[],Double[])` | Computes the DPO loss for a batch of preference pairs. |
| `ComputePreferenceAccuracy(Double[],Double[],Double[],Double[])` | Computes the accuracy of the model's implicit preference on a batch. |
| `ComputeRewardMargins(Double[],Double[],Double[],Double[])` | Computes the implicit reward margin for each preference pair. |

