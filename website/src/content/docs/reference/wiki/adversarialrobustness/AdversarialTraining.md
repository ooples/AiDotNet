---
title: "AdversarialTraining<T, TInput, TOutput>"
description: "Implements adversarial training as a defense mechanism."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Defenses`

Implements adversarial training as a defense mechanism.

## For Beginners

Adversarial training is like vaccinating your model.
Just as vaccines expose your immune system to weakened pathogens so it learns to fight them,
adversarial training exposes your model to adversarial examples during training so it learns
to resist them. This is one of the most effective defenses against adversarial attacks.

## How It Works

Adversarial training augments the training data with adversarial examples,
teaching the model to correctly classify both clean and adversarial inputs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AdversarialTraining(AdversarialDefenseOptions<>)` | Initializes a new instance of adversarial training. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Engine` | Gets the global execution engine for vectorized operations. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Apply1DDCTQuantization(Vector<>)` | 1D DCT-quantize-inverse-DCT fallback for non-square or sub-block inputs. |
| `ApplyDefense([],[],IFullModel<,,>)` |  |
| `Dct2D(Double[0:,0:])` | 2D type-II DCT (the JPEG DCT) computed via the separable two-pass form over an 8×8 block. |
| `Deserialize(Byte[])` |  |
| `EvaluateRobustness(IFullModel<,,>,[],[],IAdversarialAttack<,,>)` |  |
| `GetOptions` |  |
| `InverseDct2D(Double[0:,0:])` | Inverse 2D DCT (IDCT) — the JPEG decode step. |
| `LoadModel(String)` |  |
| `PreprocessInput()` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |

