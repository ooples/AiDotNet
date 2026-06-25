---
title: "GradientCommitmentData<T>"
description: "Contains the data for a gradient commitment."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Contains the data for a gradient commitment.

## Properties

| Property | Summary |
|:-----|:--------|
| `ClientId` | Gets or sets the client ID that created this commitment. |
| `CommitmentValue` | Gets or sets the commitment value (cryptographic hash or group element). |
| `Gradient` | Gets or sets the committed gradient tensor (null until opened). |
| `Randomness` | Gets or sets the randomness used to create the commitment (kept secret until open). |
| `Round` | Gets or sets the round number when the commitment was created. |

