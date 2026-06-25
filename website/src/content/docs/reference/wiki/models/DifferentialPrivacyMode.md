---
title: "DifferentialPrivacyMode"
description: "Specifies where differential privacy noise is applied in the federated learning pipeline."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies where differential privacy noise is applied in the federated learning pipeline.

## How It Works

**For Beginners:** Differential privacy can be applied at different points:

- Local DP: each client adds noise before sending updates (stronger protection vs server).
- Central DP: the server adds noise after aggregation (simpler and often higher utility).
- Both: apply local and central DP for defense-in-depth.

## Fields

| Field | Summary |
|:-----|:--------|
| `Central` | Apply noise on the server after aggregation. |
| `Local` | Apply noise on clients before sending updates. |
| `LocalAndCentral` | Apply both local and central differential privacy. |
| `None` | No differential privacy is applied. |
| `Shuffle` | Shuffle model DP: clients add local noise, a shuffler permutes updates before the server sees them, achieving central-DP-level accuracy with local-DP trust. |

