---
title: "VerificationLevel"
description: "Specifies the level of zero-knowledge verification applied to client updates."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the level of zero-knowledge verification applied to client updates.

## For Beginners

Higher verification levels provide stronger guarantees
but cost more computation time. Choose based on your threat model:

## Fields

| Field | Summary |
|:-----|:--------|
| `CommitmentOnly` | Commitment only — clients commit before aggregation to prevent adaptive attacks. |
| `ElementBound` | Element bound — clients prove each gradient component is within [-B, B]. |
| `FullComputation` | Full computation — clients prove the entire training computation was correct. |
| `LossThreshold` | Loss threshold — clients prove local loss is below a threshold. |
| `None` | No verification — trust all clients. |
| `NormBound` | Norm bound — clients prove gradient L2 norm is within [0, C]. |

