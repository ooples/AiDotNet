---
title: "LabelDifferentialPrivacy<T>"
description: "Implements differential privacy protection for label holder gradients in VFL."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Implements differential privacy protection for label holder gradients in VFL.

## For Beginners

In VFL, the label holder (e.g., a hospital knowing patient outcomes)
sends gradients back to feature parties (e.g., a bank knowing income). Without protection,
the bank could analyze gradient patterns to figure out which patients had bad outcomes.

## How It Works

This class adds calibrated Gaussian noise to the gradients before they're shared.
The noise is carefully sized so that even a sophisticated attacker cannot reliably distinguish
whether a specific individual's data was used, providing differential privacy guarantees.

**Privacy accounting:** Uses the Gaussian mechanism with Renyi Differential Privacy (RDP)
composition to track cumulative privacy loss across epochs. When the budget is exhausted,
training must stop.

**Reference:** Abadi et al., "Deep Learning with Differential Privacy", ACM CCS 2016.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LabelDifferentialPrivacy(Double,Double,Double,Nullable<Int32>)` | Initializes a new instance of `LabelDifferentialPrivacy`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddGaussianNoise(Tensor<>,Double)` | Adds Gaussian noise with the specified standard deviation to each element. |
| `ClipGradientNorm(Tensor<>,Double)` | Clips the gradient tensor so its L2 norm does not exceed the specified maximum. |
| `GetPrivacyBudgetSpent` |  |
| `ProtectGradients(Tensor<>)` |  |
| `ProtectLoss()` |  |
| `SampleGaussian(Double,Double)` | Samples from a Gaussian distribution using the Box-Muller transform. |
| `SampleLaplace(Double)` | Samples from a Laplace distribution. |

