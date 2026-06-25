---
title: "AutoAttack<T, TInput, TOutput>"
description: "Implements the AutoAttack framework - an ensemble of diverse attacks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Attacks`

Implements the AutoAttack framework - an ensemble of diverse attacks.

## For Beginners

AutoAttack is like having multiple expert attackers work together.
Instead of using just one attack method, it runs several different attacks and picks the
best result. This makes it very thorough and hard to defend against - if your model can
resist AutoAttack, it's genuinely robust!

## How It Works

AutoAttack combines multiple attack methods to provide a reliable evaluation of
adversarial robustness without manual parameter tuning.

This implementation includes:

- PGD (Projected Gradient Descent)
- C and W (Carlini and Wagner)
- FGSM (Fast Gradient Sign Method)

Original paper: "Reliable evaluation of adversarial robustness with an ensemble of diverse
parameter-free attacks" by Croce and Hein (2020)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AutoAttack(AdversarialAttackOptions<>)` | Initializes a new instance of AutoAttack. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` |  |
| `GenerateAdversarialExample(,,IFullModel<,,>)` | Generates an adversarial example using the AutoAttack ensemble. |
| `GetClassIndex(Vector<>)` | Gets the class index from a label vector (argmax for one-hot or probability vectors). |
| `IsSuccessfulAttack(Vector<>,Int32)` | Checks if an attack was successful based on the model's output. |

