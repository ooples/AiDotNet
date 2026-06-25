---
title: "FederatedPrivacyAccountant"
description: "Specifies which privacy accountant to use for reporting privacy spend in federated learning."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies which privacy accountant to use for reporting privacy spend in federated learning.

## How It Works

**For Beginners:** A privacy accountant tracks how much privacy budget has been spent over rounds.
Different accountants can report tighter (less pessimistic) bounds depending on assumptions.

## Fields

| Field | Summary |
|:-----|:--------|
| `Basic` | Basic composition of (ε, δ) over rounds. |
| `Rdp` | Rényi Differential Privacy (RDP) accounting (recommended when using Gaussian mechanisms). |

