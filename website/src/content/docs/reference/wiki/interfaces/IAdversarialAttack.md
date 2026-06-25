---
title: "IAdversarialAttack<T, TInput, TOutput>"
description: "Defines the contract for adversarial attack algorithms that generate adversarial examples."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for adversarial attack algorithms that generate adversarial examples.

## How It Works

An adversarial attack crafts inputs that cause machine learning models to make mistakes,
used for robustness testing and improving model security.

**For Beginners:** Think of an adversarial attack as a "stress test" for your AI model.
Just like testing if a building can withstand an earthquake, these attacks test if your model
can handle tricky inputs that are designed to fool it.

Common examples of adversarial attacks include:

- FGSM (Fast Gradient Sign Method): Quick attacks using gradient information
- PGD (Projected Gradient Descent): More powerful iterative attacks
- C&W (Carlini & Wagner): Sophisticated optimization-based attacks

Why adversarial attacks matter:

- They reveal vulnerabilities in models before deployment
- They help create more robust models through adversarial training
- They're essential for safety-critical applications (self-driving cars, medical diagnosis)
- They demonstrate potential security risks

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` | Calculates the perturbation added to create an adversarial example. |
| `GenerateAdversarialBatch([],[],IFullModel<,,>)` | Generates a batch of adversarial examples from multiple clean inputs. |
| `GenerateAdversarialExample(,,IFullModel<,,>)` | Generates adversarial examples from clean input data. |
| `GetOptions` | Gets the configuration options for the adversarial attack. |
| `Reset` | Resets the attack state to prepare for a fresh attack run. |

