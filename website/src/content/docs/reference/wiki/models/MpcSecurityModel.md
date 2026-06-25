---
title: "MpcSecurityModel"
description: "Specifies the adversary model assumed by the MPC protocol."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the adversary model assumed by the MPC protocol.

## For Beginners

The security model describes how much we trust the participants:

## Fields

| Field | Summary |
|:-----|:--------|
| `CovertSecurity` | Covert — cheating is detected with a configurable probability. |
| `Malicious` | Malicious — parties may deviate arbitrarily. |
| `SemiHonest` | Semi-honest (honest-but-curious) — parties follow the protocol but try to infer extra info. |

