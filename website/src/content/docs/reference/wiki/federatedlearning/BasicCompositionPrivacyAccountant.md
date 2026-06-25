---
title: "BasicCompositionPrivacyAccountant"
description: "Privacy accountant using basic (naive) composition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Privacy.Accounting`

Privacy accountant using basic (naive) composition.

## How It Works

**For Beginners:** Basic composition simply adds up privacy spend across rounds:

- epsilon_total = sum(epsilon_round)
- delta_total = sum(delta_round)

This is simple but can be pessimistic compared to tighter accountants.

