---
title: "IAlignmentMethod<T>"
description: "Defines the contract for AI alignment methods that ensure models behave according to human values and intentions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for AI alignment methods that ensure models behave according to human values and intentions.

## How It Works

AI alignment focuses on making AI systems that reliably do what humans want them to do,
even in novel situations where their behavior wasn't explicitly programmed.

**For Beginners:** Think of AI alignment as "teaching good behavior" to AI systems.
Just like teaching children values and ethics so they make good decisions on their own,
alignment methods help AI systems understand and follow human intentions.

Common alignment approaches include:

- RLHF (Reinforcement Learning from Human Feedback): Train models using human preferences
- Constitutional AI: Teach models principles to guide their behavior
- Red Teaming: Systematically test for harmful or unintended behaviors

Why AI alignment matters:

- Prevents models from pursuing goals in harmful ways
- Ensures models are helpful, harmless, and honest
- Critical for deploying powerful AI systems safely
- Helps models generalize human values to new situations

## Methods

| Method | Summary |
|:-----|:--------|
| `AlignModel(IPredictiveModel<,Vector<>,Vector<>>,AlignmentFeedbackData<>)` | Aligns a model using feedback from human evaluators or preferences. |
| `ApplyConstitutionalPrinciples(IPredictiveModel<,Vector<>,Vector<>>,String[])` | Applies constitutional principles to guide model behavior. |
| `EvaluateAlignment(IPredictiveModel<,Vector<>,Vector<>>,AlignmentEvaluationData<>)` | Evaluates how well a model is aligned with human values. |
| `GetOptions` | Gets the configuration options for the alignment method. |
| `PerformRedTeaming(IPredictiveModel<,Vector<>,Vector<>>,Matrix<>)` | Performs red teaming to identify potential misalignment or harmful behaviors. |
| `Reset` | Resets the alignment method state. |

