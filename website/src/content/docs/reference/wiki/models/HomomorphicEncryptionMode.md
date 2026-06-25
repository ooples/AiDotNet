---
title: "HomomorphicEncryptionMode"
description: "Specifies how homomorphic encryption is applied during federated aggregation."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies how homomorphic encryption is applied during federated aggregation.

## How It Works

**For Beginners:** HE-only encrypts everything, while hybrid encrypts only selected parameters to reduce cost.

## Fields

| Field | Summary |
|:-----|:--------|
| `HeOnly` | Encrypt all parameters for aggregation. |
| `Hybrid` | Encrypt only selected parameter ranges; remaining parameters use the normal pipeline. |

