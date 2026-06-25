---
title: "CWAttack<T, TInput, TOutput>"
description: "Implements the Carlini and Wagner (C and W) attack."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Attacks`

Implements the Carlini and Wagner (C and W) attack.

## For Beginners

C and W is one of the most sophisticated attacks. Instead of
following gradients, it treats creating adversarial examples as a carefully crafted
optimization problem. It's slower than FGSM or PGD but often finds adversarial examples
that are more subtle and harder to defend against.

## How It Works

C and W is an optimization-based attack that formulates adversarial example generation as
an optimization problem, typically producing stronger attacks than gradient-based methods.

Original paper: "Towards Evaluating the Robustness of Neural Networks"
by Carlini and Wagner (2017)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CWAttack(AdversarialAttackOptions<>)` | Initializes a new instance of the C and W attack. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` |  |
| `CloneVector(Vector<>)` | Clones a vector using vectorized operations. |
| `ComputeAttackLoss(Vector<>,Int32)` | Computes the attack loss for C and W. |
| `ComputeAttackLossGradient(Vector<>,Int32)` | Computes the gradient of the attack loss with respect to the model output. |
| `ComputeFiniteDifferenceGradient(Double[],Vector<>,Int32,Double,,IFullModel<,,>)` | Computes the gradient using finite-difference approximation as a fallback. |
| `ComputeObjective(Double[],Vector<>,Vector<>,Int32,Double)` | Computes the objective value for a given w. |
| `ComputeObjectiveAndGradient(Double[],Vector<>,Vector<>,Int32,Double,,IFullModel<,,>)` | Computes the objective function and its gradient. |
| `GenerateAdversarialExample(,,IFullModel<,,>)` | Generates an adversarial example using the C and W L2 attack. |
| `GetClassIndex(Vector<>)` | Gets the class index from a label vector (argmax for one-hot or probability vectors). |
| `IsSuccessfulAttack(Vector<>,Int32)` | Checks if the attack was successful. |
| `TanhSpaceToInputSpace(Double[])` | Converts from tanh space (w) to valid input space [0, 1]. |

