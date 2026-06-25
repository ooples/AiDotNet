---
title: "ViTAdversarialAttack<T, TInput, TOutput>"
description: "Implements Vision Transformer (ViT)-specific adversarial attacks that exploit the self-attention mechanism and patch-based architecture."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Attacks`

Implements Vision Transformer (ViT)-specific adversarial attacks that exploit the
self-attention mechanism and patch-based architecture.

## For Beginners

Vision Transformers look at images in small patches and use
"attention" to decide which patches matter most. This attack finds the most important
patches and makes tiny changes to them, which is more effective than changing random
pixels. It's like knowing which cards matter in a card game and only swapping those.

## How It Works

Unlike CNN-targeted attacks (FGSM, PGD) that rely on local gradient information,
ViT attacks exploit the global self-attention mechanism. This attack perturbs patches
that the model attends to most, generating adversarial examples that are more effective
against ViT architectures while using smaller perturbation budgets.

**References:**

- On the adversarial robustness of Vision Transformers (Bhojanapalli et al., 2021)
- ViT adversarial robustness analysis and improved training (CVPR 2024)
- Towards Robust Vision Transformer (Mo et al., 2022)

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ViTAdversarialAttack(AdversarialAttackOptions<>,Int32,Int32,Int32)` | Initializes a new ViT-specific adversarial attack. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculatePerturbation(,)` |  |
| `GenerateAdversarialExample(,,IFullModel<,,>)` |  |

