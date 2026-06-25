---
title: "IFineTuning<T, TInput, TOutput>"
description: "Defines the contract for fine-tuning methods that adapt pre-trained models to specific tasks or preferences."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for fine-tuning methods that adapt pre-trained models to specific tasks or preferences.

## For Beginners

Fine-tuning is like specialized training for an AI that already knows the basics.
Just like a doctor goes through general education before specializing, AI models first learn general knowledge
(pre-training) and then learn specific skills or behaviors (fine-tuning).

## How It Works

Fine-tuning encompasses a wide range of techniques for adapting models, from supervised fine-tuning (SFT)
to advanced preference optimization methods like DPO, RLHF, and their variants.

**Categories of Fine-Tuning Methods:**

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Gets the category of this fine-tuning method. |
| `MethodName` | Gets the name of this fine-tuning method. |
| `RequiresReferenceModel` | Gets whether this method requires a reference model. |
| `RequiresRewardModel` | Gets whether this method requires a reward model. |
| `SupportsPEFT` | Gets whether this method supports parameter-efficient fine-tuning (PEFT). |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` | Evaluates the fine-tuning quality of a model. |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` | Fine-tunes a model using the configured method and provided training data. |
| `GetOptions` | Gets the configuration options for this fine-tuning method. |
| `Reset` | Resets the fine-tuning method state. |

