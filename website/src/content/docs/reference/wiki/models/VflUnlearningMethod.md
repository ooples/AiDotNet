---
title: "VflUnlearningMethod"
description: "Specifies the method used to remove an entity's influence from a trained VFL model."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the method used to remove an entity's influence from a trained VFL model.

## For Beginners

Under GDPR and similar regulations, individuals have the "right to
be forgotten". If a patient asks to be removed from a model, the model must be updated so that
it no longer contains any information learned from that patient's data. These methods vary
in how thoroughly they remove the influence and how computationally expensive they are.

## Fields

| Field | Summary |
|:-----|:--------|
| `Certified` | Certified unlearning with mathematical guarantees that the unlearned model is statistically indistinguishable from a model trained without the removed data. |
| `GradientAscent` | Apply gradient ascent on the removed entities to approximately reverse their influence. |
| `PrimalDual` | Uses a primal-dual optimization framework for both sample and label unlearning in VFL. |
| `Retraining` | Retrain the model from scratch without the removed entities. |

