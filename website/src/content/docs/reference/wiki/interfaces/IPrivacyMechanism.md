---
title: "IPrivacyMechanism<TModel>"
description: "Defines privacy-preserving mechanisms for federated learning to protect client data."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines privacy-preserving mechanisms for federated learning to protect client data.

## How It Works

This interface represents techniques to ensure that model updates don't leak sensitive
information about individual data points in clients' local datasets.

**For Beginners:** Privacy mechanisms are like filters that protect sensitive information
while still allowing useful knowledge to be shared.

Think of privacy mechanisms as protective measures:

- Differential Privacy: Adds carefully calibrated noise to make individual data unidentifiable
- Secure Aggregation: Encrypts updates so the server only sees the combined result
- Homomorphic Encryption: Allows computation on encrypted data

For example, in a hospital scenario:

- Without privacy: Model updates might reveal information about specific patients
- With differential privacy: Random noise is added so you can't identify individual patients
- The noise is calibrated so the overall patterns remain accurate

Privacy mechanisms provide mathematical guarantees:

- Epsilon (ε): Privacy budget - lower values mean stronger privacy
- Delta (δ): Probability that privacy guarantee fails
- Common setting: ε=1.0, δ=1e-5 means strong privacy with high confidence

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyPrivacy(,Double,Double)` | Applies privacy-preserving techniques to a model update before sharing it. |
| `GetMechanismName` | Gets the name of the privacy mechanism. |
| `GetPrivacyBudgetConsumed` | Gets the current privacy budget consumed by this mechanism. |

