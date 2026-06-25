---
title: "AdversarialAttackOptions<T>"
description: "Configuration options for adversarial attack algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for adversarial attack algorithms.

## For Beginners

These settings control how the "stress test" for your AI works.
You can adjust how strong the attacks are, how many attempts they make to fool your model,
and what type of changes they're allowed to make to inputs.

## How It Works

These options control how adversarial examples are generated, including the strength
of perturbations, attack iterations, and norm constraints.

## Properties

| Property | Summary |
|:-----|:--------|
| `Epsilon` | Gets or sets the maximum perturbation budget (epsilon). |
| `IsTargeted` | Gets or sets whether to use targeted or untargeted attacks. |
| `Iterations` | Gets or sets the number of iterations for iterative attacks. |
| `NormType` | Gets or sets the norm type for perturbation constraints. |
| `RandomSeed` | Gets or sets the random seed for reproducibility. |
| `StepSize` | Gets or sets the step size for iterative attacks. |
| `TargetClass` | Gets or sets the target class for targeted attacks. |
| `UseRandomStart` | Gets or sets whether to use random initialization. |

