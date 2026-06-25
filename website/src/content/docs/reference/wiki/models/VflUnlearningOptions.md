---
title: "VflUnlearningOptions"
description: "Configuration for GDPR-compliant entity unlearning in vertical federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration for GDPR-compliant entity unlearning in vertical federated learning.

## For Beginners

Under GDPR and similar privacy regulations, individuals have the
"right to be forgotten". When a person requests deletion, not only must their data be removed
from storage, but the model must also be updated to remove any influence of their data.
This is called "machine unlearning".

## How It Works

In VFL, this is more complex than in standard ML because data is spread across
multiple parties, and the model is split across parties. Each party must participate
in the unlearning process.

Example:

## Properties

| Property | Summary |
|:-----|:--------|
| `CertificationEpsilon` | Gets or sets the privacy budget (epsilon) for certified unlearning. |
| `Enabled` | Gets or sets whether unlearning support is enabled for this VFL training run. |
| `GradientAscentSteps` | Gets or sets the number of gradient ascent steps for the GradientAscent method. |
| `MaxUnlearnBatchSize` | Gets or sets the maximum number of entities to unlearn in a single batch. |
| `Method` | Gets or sets the unlearning method to use. |
| `UnlearningLearningRate` | Gets or sets the learning rate for gradient ascent unlearning. |
| `VerifyUnlearning` | Gets or sets whether to verify unlearning effectiveness by checking that the unlearned model cannot distinguish removed entities from unseen entities. |

