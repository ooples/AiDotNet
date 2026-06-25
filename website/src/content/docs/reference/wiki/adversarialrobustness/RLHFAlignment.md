---
title: "RLHFAlignment<T>"
description: "Implements Reinforcement Learning from Human Feedback (RLHF) for AI alignment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Alignment`

Implements Reinforcement Learning from Human Feedback (RLHF) for AI alignment.

## For Beginners

RLHF is like having a human teacher grade the AI's responses
and using those grades to improve the AI. The AI learns what humans prefer and adjusts
its behavior accordingly. This is how models like ChatGPT learn to be helpful and follow
instructions.

## How It Works

RLHF trains models to align with human preferences by learning a reward model
from human feedback and using it to fine-tune the model via reinforcement learning.

Original approaches: "Learning to summarize from human feedback" (OpenAI, 2020),
"Training language models to follow instructions with human feedback" (InstructGPT, 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RLHFAlignment(AlignmentMethodOptions<>)` | Initializes a new instance of RLHF alignment. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsRewardModelTrained` | Gets whether the reward model has been trained. |
| `StrictHonestyMode` | When false (default — industry convention: honesty-on-non-comparable), `Vector{` returns `true` for inputs the heuristic cannot evaluate (null / empty / length-mismatch input vs output) and emits a Trace warning so the unscored pair is obse… |

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignModel(IPredictiveModel<,Vector<>,Vector<>>,AlignmentFeedbackData<>)` |  |
| `ApplyConstitutionalPrinciples(IPredictiveModel<,Vector<>,Vector<>>,String[])` |  |
| `Deserialize(Byte[])` |  |
| `EvaluateAlignment(IPredictiveModel<,Vector<>,Vector<>>,AlignmentEvaluationData<>)` |  |
| `GetOptions` |  |
| `LoadModel(String)` |  |
| `PerformRedTeaming(IPredictiveModel<,Vector<>,Vector<>>,Matrix<>)` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

