---
title: "ReinforcementLearningHumanFeedback<T, TInput, TOutput>"
description: "Implements Reinforcement Learning from Human Feedback (RLHF) with PPO for fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Reinforcement Learning from Human Feedback (RLHF) with PPO for fine-tuning.

## For Beginners

RLHF is like training a model with a coach (reward model)
that tells it how good its responses are. The model learns to generate responses
that the coach rates highly, while not straying too far from its original behavior.

## How It Works

RLHF is the foundational approach for aligning language models with human preferences.
It uses a reward model trained on human feedback to guide policy optimization via PPO.

RLHF pipeline:

1. Train a reward model on human preference data
2. Use PPO to optimize the policy against the reward model
3. Add KL penalty to prevent reward hacking

Original paper: "Training language models to follow instructions with human feedback"
by Ouyang et al. (2022) - InstructGPT

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ReinforcementLearningHumanFeedback(FineTuningOptions<>)` | Initializes a new instance of RLHF fine-tuning with PPO. |

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
| `CollectExperience(IFullModel<,,>,FineTuningData<,,>)` | Collects experience (states, actions, rewards) from the policy. |
| `ComputeAdvantages(List<ReinforcementLearningHumanFeedback<,,>.PPOExperience>,Double,Double)` | Computes advantages using Generalized Advantage Estimation (GAE). |
| `ComputeEntropyLoss(IFullModel<,,>,)` | Computes entropy loss to encourage exploration. |
| `ComputePPOLossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,List<ReinforcementLearningHumanFeedback<,,>.PPOExperience>,Double,Double,Double,Double,CancellationToken)` | Computes PPO loss and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `SetRewardFunction(Func<,,Double>)` | Sets the reward function (or reward model) for evaluating responses. |
| `SetValueModel(IFullModel<,,>)` | Sets the value model for PPO advantage estimation. |

