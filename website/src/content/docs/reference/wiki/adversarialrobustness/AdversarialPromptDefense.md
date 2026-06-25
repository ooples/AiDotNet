---
title: "AdversarialPromptDefense<T, TInput, TOutput>"
description: "Implements adaptive visual prompt-based defense that prepends learned perturbation-resistant tokens/patches to inputs, improving adversarial robustness without model retraining."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Defenses`

Implements adaptive visual prompt-based defense that prepends learned perturbation-resistant
tokens/patches to inputs, improving adversarial robustness without model retraining.

## For Beginners

Imagine putting a special protective "frame" around every image before
showing it to the AI. This frame is carefully designed to cancel out any malicious changes an
attacker might have made. The AI itself doesn't need to change — the frame does all the
protective work. This is much faster to train than retraining the whole model.

## How It Works

Visual prompt defense adds a set of learned "defense tokens" (for ViTs) or border patches
(for CNNs) to each input before inference. These prompts are optimized to absorb adversarial
perturbations, effectively neutralizing attacks. The key advantage is that the base model
weights remain frozen — only the prompt vectors are trained, making this defense extremely
lightweight and model-agnostic.

**References:**

- RobustPrompt: Adaptive visual prompts, 61.1% improvement vs PGD (2025)
- Visual Prompting for Adversarial Robustness (Chen et al., ICASSP 2024)
- Adversarial Visual Prompt Tuning (Fu et al., 2023)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialPromptDefense(AdversarialDefenseOptions<>,Int32,Double,Int32)` | Initializes a new visual prompt-based adversarial defense. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefense([],[],IFullModel<,,>)` |  |
| `Deserialize(Byte[])` |  |
| `EvaluateRobustness(IFullModel<,,>,[],[],IAdversarialAttack<,,>)` |  |
| `GetOptions` |  |
| `LoadModel(String)` |  |
| `PreprocessInput()` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

