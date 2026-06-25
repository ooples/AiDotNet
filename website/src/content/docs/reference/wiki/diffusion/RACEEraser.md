---
title: "RACEEraser<T>"
description: "RACE: Robust Adversarial Concept Erasure for removing concepts resilient to red-teaming attacks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Diffusion.Safety`

RACE: Robust Adversarial Concept Erasure for removing concepts resilient to red-teaming attacks.

## For Beginners

Standard concept erasure can sometimes be "tricked" by cleverly
worded prompts that recover the erased content. RACE prevents this by training with an
attacker and defender at the same time. The attacker tries to find prompts that bypass
the erasure, and the defender learns to block those too. It's like training a security
system by constantly testing it with new attack strategies.

## How It Works

RACE addresses the vulnerability of standard concept erasure methods to adversarial prompts
that can recover erased concepts. It uses an adversarial training loop: a red-team module
generates challenging prompts trying to recover the erased concept, while the erasure module
learns to be robust against these attacks. This min-max game produces erasure that resists
prompt-based attacks.

Reference: Pham et al., "Robust Concept Erasure via Adversarial Training (RACE)", 2024

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RACEEraser(Double,Double,Double,Int32,Int32)` | Initializes a new RACE eraser. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AdversarialWeight` | Gets the adversarial weight. |
| `AttackLearningRate` | Gets the attack learning rate. |
| `ErasureLearningRate` | Gets the erasure learning rate. |
| `InnerSteps` | Gets the number of inner adversarial steps. |
| `OuterSteps` | Gets the number of outer erasure steps. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AttackerStep(Vector<>,Vector<>)` | Applies a single adversarial perturbation step to a prompt embedding. |
| `ComputeAttackerLoss(Vector<>,Vector<>)` | Computes the attacker's loss (red-team objective, maximized during inner loop). |
| `ComputeDefenderLoss(Vector<>,Vector<>,Vector<>,Vector<>)` | Computes the adversarial erasure loss (defender's loss). |

