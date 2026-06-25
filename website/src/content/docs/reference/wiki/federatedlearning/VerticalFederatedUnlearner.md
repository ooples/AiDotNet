---
title: "VerticalFederatedUnlearner<T>"
description: "Implements GDPR-compliant entity unlearning for vertical federated learning models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Vertical`

Implements GDPR-compliant entity unlearning for vertical federated learning models.

## For Beginners

Under GDPR and similar regulations, individuals have the "right to
be forgotten". When someone requests deletion, not only must their data be removed from storage,
but the model must also be updated to remove any influence of their data. This is called
"machine unlearning".

## How It Works

In VFL, unlearning is more complex than in standard ML because:

**Methods supported:**

**Reference:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `VerticalFederatedUnlearner(VflUnlearningOptions)` | Initializes a new instance of `VerticalFederatedUnlearner`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddCertificationNoise(IReadOnlyList<Tensor<>>,Double,Nullable<Int32>)` | Adds calibrated noise to model parameters for certified unlearning. |
| `ComputeInfluence(Tensor<>,Tensor<>,Tensor<>)` | Computes the influence of a set of entities on the model parameters. |
| `VerifyUnlearning(Tensor<>,Tensor<>,Tensor<>)` | Verifies that unlearning was effective by checking if the model's behavior on the unlearned entities is statistically indistinguishable from a model that was never trained on those entities. |

