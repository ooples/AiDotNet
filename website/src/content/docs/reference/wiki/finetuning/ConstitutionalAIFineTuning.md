---
title: "ConstitutionalAIFineTuning<T, TInput, TOutput>"
description: "Implements Constitutional AI (CAI) fine-tuning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FineTuning`

Implements Constitutional AI (CAI) fine-tuning.

## For Beginners

CAI is like giving the model a rulebook to follow.
The model learns to check its own answers against rules like "be helpful" or
"don't cause harm" and revise answers that break the rules.

## How It Works

Constitutional AI uses a set of principles (a "constitution") to guide model behavior.
The model learns to critique and revise its own outputs based on these principles.

CAI training has two phases:

1. Supervised Learning from AI Feedback (SLAIF): Generate, critique, revise
2. Reinforcement Learning from AI Feedback (RLAIF): Use revised outputs for preference training

Original paper: "Constitutional AI: Harmlessness from AI Feedback"
by Bai et al. (2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConstitutionalAIFineTuning(FineTuningOptions<>)` | Initializes a new instance of Constitutional AI fine-tuning. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` |  |
| `MethodName` |  |
| `Principles` | Gets the constitutional principles used for training. |
| `RequiresReferenceModel` |  |
| `RequiresRewardModel` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCAILossAndUpdateAsync(IFullModel<,,>,FineTuningData<,,>,Double,Int32,Double,CancellationToken)` | Computes the CAI loss for a batch and updates model parameters. |
| `EvaluateAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `EvaluateConstitutionalCompliance(,)` | Evaluates how well an output complies with constitutional principles. |
| `FineTuneAsync(IFullModel<,,>,FineTuningData<,,>,CancellationToken)` |  |
| `ValidateTrainingData(FineTuningData<,,>)` |  |

