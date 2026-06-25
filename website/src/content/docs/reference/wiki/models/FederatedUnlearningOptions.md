---
title: "FederatedUnlearningOptions"
description: "Configuration options for federated unlearning (right to be forgotten)."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for federated unlearning (right to be forgotten).

## For Beginners

These options control how the system removes a client's contribution
from the trained model when they exercise their right to be forgotten (GDPR, CCPA, LGPD).

## Properties

| Property | Summary |
|:-----|:--------|
| `InfluenceTolerance` | Gets or sets the tolerance for influence function convergence. |
| `MaxInfluenceIterations` | Gets or sets the maximum number of iterations for influence function computation. |
| `MaxUnlearningEpochs` | Gets or sets the maximum epochs for approximate unlearning methods. |
| `Method` | Gets or sets the unlearning method. |
| `NoiseScale` | Gets or sets the noise scale for diffusive noise unlearning. |
| `UnlearningLearningRate` | Gets or sets the learning rate for gradient ascent unlearning. |
| `VerificationEnabled` | Gets or sets whether to verify unlearning correctness. |
| `VerificationThreshold` | Gets or sets the verification threshold: maximum cosine similarity between the unlearned model and a model trained with the target client that is considered acceptable. |

