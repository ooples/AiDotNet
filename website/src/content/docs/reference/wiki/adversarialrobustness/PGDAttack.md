---
title: "PGDAttack<T, TInput, TOutput>"
description: "Implements the Projected Gradient Descent (PGD) attack."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Attacks`

Implements the Projected Gradient Descent (PGD) attack.

## For Beginners

PGD is like FGSM but repeated multiple times with smaller steps.
Instead of one big jump, it takes many small steps, checking after each step to make sure
it hasn't gone too far. This makes it much more powerful than FGSM but also slower.

## How It Works

PGD is an iterative variant of FGSM that applies multiple small perturbation steps,
projecting back into the allowed perturbation region after each step.

PGD is considered one of the strongest first-order adversarial attacks and is commonly
used for adversarial training and robustness evaluation.

Original paper: "Towards Deep Learning Models Resistant to Adversarial Attacks"
by Madry et al. (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PGDAttack(AdversarialAttackOptions<>)` | Initializes a new instance of the PGD attack. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` |  |
| `CloneVector(Vector<>)` | Clones a vector using vectorized operations. |
| `ComputeMseLoss(Vector<>,Vector<>)` | Computes the mean squared error loss between output and target vectors. |
| `ComputeNumericalGradient(Vector<>,Vector<>,,IFullModel<,,>)` | Computes the gradient of MSE loss w.r.t. |
| `ComputeTapeGradient(Vector<>,Vector<>,NeuralNetworkBase<>)` | Computes exact gradient of loss w.r.t. |
| `GenerateAdversarialExample(,,IFullModel<,,>)` | Generates an adversarial example using the PGD attack. |
| `ProjectToEpsilonBall(Vector<>,Vector<>,)` | Projects the adversarial example back into the epsilon-ball around the original input. |
| `RandomStartingPoint(Vector<>,)` | Generates a random starting point within the epsilon-ball. |

