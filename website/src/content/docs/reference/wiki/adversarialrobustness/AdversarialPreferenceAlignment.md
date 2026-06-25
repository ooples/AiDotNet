---
title: "AdversarialPreferenceAlignment<T>"
description: "Implements adversarial preference alignment that combines RLHF with adversarial robustness, ensuring the model maintains alignment properties even under adversarial perturbation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Defenses`

Implements adversarial preference alignment that combines RLHF with adversarial robustness,
ensuring the model maintains alignment properties even under adversarial perturbation.

## For Beginners

RLHF teaches an AI to give good responses. But an attacker can craft
tricky inputs that bypass this training. This module trains the AI to give good responses
EVEN when the input has been tampered with — like teaching a student to follow school rules
even when other students try to distract or trick them.

## How It Works

Standard RLHF alignment can be broken by adversarial inputs that shift model behavior
away from aligned responses. This defense augments the RLHF training loop with adversarial
examples, teaching the reward model and policy to maintain preference alignment even when
inputs are adversarially perturbed. The result is a model that remains helpful, harmless,
and honest even under attack.

**References:**

- Adversarial RLHF: Adversarially robust alignment (2024)
- Safety-Tuned LLaMAs: Lessons from improving safety of LLMs (2024)
- Robustness of RLHF alignment under adversarial prompts (NAACL 2025)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialPreferenceAlignment(AlignmentMethodOptions<>,Double,Double,Int32)` | Initializes a new adversarial preference alignment module. |

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

