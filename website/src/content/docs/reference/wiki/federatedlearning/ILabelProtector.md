---
title: "ILabelProtector<T>"
description: "Protects label holder information from being inferred by feature-holding parties."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.Vertical`

Protects label holder information from being inferred by feature-holding parties.

## For Beginners

In vertical FL, the label holder (e.g., a hospital that knows
patient outcomes) computes the loss and sends gradients back to feature parties (e.g., a bank
that knows income). Without protection, the bank could analyze these gradients to figure out
which patients had bad outcomes.

## How It Works

Label protection adds noise or other protections to the gradients before they're sent
to feature parties, preventing this kind of inference attack. The trade-off is between
privacy (more noise = more protection) and accuracy (more noise = slower learning).

**Common attacks prevented:**

## Methods

| Method | Summary |
|:-----|:--------|
| `GetPrivacyBudgetSpent` | Gets the cumulative privacy budget consumed so far. |
| `ProtectGradients(Tensor<>)` | Adds privacy protection to gradients before sending them to feature-holding parties. |
| `ProtectLoss()` | Adds privacy protection to the loss value before sharing it. |

