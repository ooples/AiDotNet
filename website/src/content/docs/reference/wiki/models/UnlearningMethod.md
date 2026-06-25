---
title: "UnlearningMethod"
description: "Specifies the federated unlearning method to use when a client requests data removal."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the federated unlearning method to use when a client requests data removal.

## For Beginners

When someone exercises their "right to be forgotten" (GDPR Article 17),
their contribution to the model must be removed. These methods differ in speed vs. guarantee strength:

## Fields

| Field | Summary |
|:-----|:--------|
| `DiffusiveNoise` | Structured noise injection targeting memorized samples (2025 research). |
| `ExactRetraining` | Retrain from scratch excluding the target client (provably correct, expensive). |
| `GradientAscent` | Gradient ascent on target client data to reverse learning (fast, approximate). |
| `InfluenceFunction` | Influence function-based removal (Newton step, efficient for small removals). |

