---
title: "GroupRelativePolicyOptimization<T, TInput, TOutput>"
description: "Implements Group Relative Policy Optimization (GRPO) for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Group Relative Policy Optimization (GRPO) for fine-tuning.

## For Beginners

GRPO is like PPO but more efficient. Instead of training a
separate critic model to estimate value, GRPO generates multiple responses for each prompt
and uses the group's average reward as the baseline. This makes training faster and
uses less GPU memory.

## How It Works

GRPO is a memory-efficient reinforcement learning algorithm developed by DeepSeek
that eliminates the need for a separate critic model, reducing memory requirements by ~50%.

Key features of GRPO:

1. No critic model needed (saves ~50% memory)
2. Group-based advantage estimation
3. Works well with verifiable rewards (RLVR)
4. Used in DeepSeekMath and DeepSeek-R1

The GRPO advantage is computed as:
A_i = (r_i - mean(r_group)) / std(r_group)
where r_group are the rewards for all responses in the group.

Original paper: "DeepSeekMath: Pushing the Limits of Mathematical Reasoning"
by Shao et al. (2024)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GroupRelativePolicyOptimization(FineTuningOptions<>)` | Initializes a new instance of GRPO fine-tuning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MethodName` |  |
| `RequiresReferenceModel` |  |
| `RequiresRewardModel` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeGRPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Int32,Double,Double,Double,CancellationToken)` | Computes the GRPO loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `SetRewardFunction(Func<,,Double>)` | Sets the reward function for evaluating responses. |

