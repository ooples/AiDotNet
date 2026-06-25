---
title: "IAdversarialDefense<T, TInput, TOutput>"
description: "Defines the contract for adversarial defense mechanisms that protect models against attacks."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for adversarial defense mechanisms that protect models against attacks.

## How It Works

An adversarial defense is a technique to make machine learning models more resistant
to adversarial attacks and improve their robustness.

**For Beginners:** Think of adversarial defenses as "armor" for your AI model.
Just like armor protects a knight from attacks, these defenses protect your model
from adversarial examples that try to fool it.

Common examples of adversarial defenses include:

- Adversarial Training: Training the model on adversarial examples to make it robust
- Input Transformations: Preprocessing inputs to remove adversarial perturbations
- Ensemble Methods: Using multiple models to make predictions more reliable

Why adversarial defenses matter:

- They make models safer for real-world deployment
- They improve model reliability under attack
- They're critical for security-sensitive applications
- They help models generalize better to unusual inputs

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyDefense([],[],IFullModel<,,>)` | Trains or hardens a model to be more resistant to adversarial attacks. |
| `EvaluateRobustness(IFullModel<,,>,[],[],IAdversarialAttack<,,>)` | Evaluates the robustness of a defended model against attacks. |
| `GetOptions` | Gets the configuration options for the adversarial defense. |
| `PreprocessInput()` | Preprocesses input data to remove or reduce adversarial perturbations. |
| `Reset` | Resets the defense state to prepare for a fresh defense application. |

