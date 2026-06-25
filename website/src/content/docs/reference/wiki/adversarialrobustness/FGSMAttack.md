---
title: "FGSMAttack<T, TInput, TOutput>"
description: "Implements the Fast Gradient Sign Method (FGSM) attack."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Attacks`

Implements the Fast Gradient Sign Method (FGSM) attack.

## For Beginners

FGSM is like finding the steepest hill and taking one big step
in that direction. It's fast but might not be as powerful as multi-step attacks like PGD.
Think of it as the "quick and dirty" attack - it's not the strongest, but it's very efficient.

## How It Works

FGSM is a simple yet effective white-box adversarial attack that uses the gradient
of the loss function to create adversarial examples in a single step.

Original paper: "Explaining and Harnessing Adversarial Examples" by Goodfellow et al. (2014)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FGSMAttack(AdversarialAttackOptions<>)` | Initializes a new instance of the FGSM attack. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` |  |
| `ComputeMseLoss(Vector<>,Vector<>)` | Computes the mean squared error loss between output and target vectors. |
| `ComputeNumericalGradient(Vector<>,Vector<>,,IFullModel<,,>)` | Computes the gradient of MSE loss w.r.t. |
| `ComputeTapeGradient(Vector<>,Vector<>,NeuralNetworkBase<>)` | Computes exact gradient of loss w.r.t. |
| `GenerateAdversarialExample(,,IFullModel<,,>)` | Generates an adversarial example using the FGSM attack. |

