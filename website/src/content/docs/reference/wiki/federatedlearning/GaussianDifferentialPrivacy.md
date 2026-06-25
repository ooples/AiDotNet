---
title: "GaussianDifferentialPrivacy<T>"
description: "GaussianDifferentialPrivacy<T> — Models & Types in AiDotNet.FederatedLearning.Privacy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy`

_No summary documentation available yet._

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `GaussianDifferentialPrivacy(Double,Nullable<Int32>)` | Initializes a new instance of the `GaussianDifferentialPrivacy` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPrivacy(Dictionary<String,[]>,Double,Double)` | Applies differential privacy to model parameters by adding calibrated Gaussian noise. |
| `CalculateL2Norm(Dictionary<String,[]>)` | Calculates the L2 norm (Euclidean norm) of all model parameters. |
| `GenerateGaussianNoise(Double,Double)` | Generates a sample from a Gaussian (normal) distribution. |
| `GetClipNorm` | Gets the gradient clipping norm used for sensitivity bounding. |
| `GetMechanismName` | Gets the name of the privacy mechanism. |
| `GetPrivacyBudgetConsumed` | Gets the total privacy budget consumed so far. |
| `ResetPrivacyBudget` | Resets the privacy budget counter. |

