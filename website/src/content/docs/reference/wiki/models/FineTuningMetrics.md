---
title: "FineTuningMetrics<T>"
description: "Metrics for evaluating fine-tuning quality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Options`

Metrics for evaluating fine-tuning quality.

## For Beginners

These metrics tell you how well the fine-tuning worked.
Lower loss is generally better, higher win rates mean the model learned preferences well,
and safety metrics ensure the model behaves appropriately.

## How It Works

This class contains metrics relevant to various fine-tuning methods:

## Properties

| Property | Summary |
|:-----|:--------|
| `AverageOutputLength` | Gets or sets the average output length. |
| `AverageReward` | Gets or sets the average reward achieved. |
| `BleuScore` | Gets or sets the BLEU score for generation quality. |
| `ChosenLogProb` | Gets or sets the chosen response log probability. |
| `CustomMetrics` | Gets or sets additional custom metrics. |
| `DistillationLoss` | Gets or sets the distillation loss. |
| `FalseRefusalRate` | Gets or sets the false refusal rate. |
| `GroupAdvantage` | Gets or sets the average group advantage. |
| `GroupRewardVariance` | Gets or sets the within-group reward variance. |
| `HarmlessnessScore` | Gets or sets the harmlessness score. |
| `HelpfulnessScore` | Gets or sets the helpfulness score. |
| `HonestyScore` | Gets or sets the honesty score. |
| `KLDivergence` | Gets or sets the KL divergence from the reference model. |
| `LogProbMargin` | Gets or sets the margin between chosen and rejected log probabilities. |
| `LossHistory` | Gets or sets the loss history over training. |
| `MethodName` | Gets or sets the fine-tuning method used. |
| `PeakMemoryGB` | Gets or sets the peak memory usage in GB. |
| `Perplexity` | Gets or sets the perplexity on validation data. |
| `PolicyEntropy` | Gets or sets the entropy of the policy. |
| `PolicyLoss` | Gets or sets the policy loss. |
| `PreferenceAccuracy` | Gets or sets the preference accuracy on held-out data. |
| `RefusalRate` | Gets or sets the refusal rate for harmful prompts. |
| `RejectedLogProb` | Gets or sets the rejected response log probability. |
| `RewardStd` | Gets or sets the reward standard deviation. |
| `RougeLScore` | Gets or sets the ROUGE-L score. |
| `TeacherAgreementRate` | Gets or sets the agreement rate with teacher model. |
| `ThroughputSamplesPerSecond` | Gets or sets the throughput in samples per second. |
| `TrainableParameters` | Gets or sets the number of trainable parameters. |
| `TrainingEndTime` | Gets or sets when the training completed. |
| `TrainingLoss` | Gets or sets the final training loss. |
| `TrainingStartTime` | Gets or sets when the training started. |
| `TrainingSteps` | Gets or sets the number of training steps completed. |
| `TrainingTimeSeconds` | Gets or sets the total training time in seconds. |
| `ValidationLoss` | Gets or sets the validation loss. |
| `ValueLoss` | Gets or sets the value function loss (for actor-critic methods). |
| `WinRate` | Gets or sets the win rate against the reference model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary` | Gets a summary of the metrics suitable for logging. |

