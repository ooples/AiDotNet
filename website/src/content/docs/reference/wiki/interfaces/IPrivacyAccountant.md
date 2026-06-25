---
title: "IPrivacyAccountant"
description: "Tracks cumulative privacy loss across federated learning rounds."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Tracks cumulative privacy loss across federated learning rounds.

## How It Works

**For Beginners:** Differential privacy has a finite "budget" (epsilon, delta).
Each training round spends some of that budget. A privacy accountant keeps track
of what was spent so you can report guarantees and enforce limits.

## Methods

| Method | Summary |
|:-----|:--------|
| `AddRound(Double,Double,Double)` | Records a single privacy event (typically one federated learning round). |
| `GetAccountantName` | Gets the name of this privacy accountant implementation. |
| `GetEpsilonAtDelta(Double)` | Gets a reported epsilon value at a given target delta (if supported by the accountant). |
| `GetTotalDeltaConsumed` | Gets the total delta consumed so far according to this accountant. |
| `GetTotalEpsilonConsumed` | Gets the total epsilon consumed so far according to this accountant. |

