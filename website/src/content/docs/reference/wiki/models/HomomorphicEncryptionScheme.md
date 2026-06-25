---
title: "HomomorphicEncryptionScheme"
description: "Specifies which homomorphic encryption scheme to use."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies which homomorphic encryption scheme to use.

## How It Works

**For Beginners:** Different HE schemes support different kinds of math:

- CKKS is best for approximate real-number arithmetic.
- BFV is best for exact integer arithmetic (often via fixed-point encoding).

## Fields

| Field | Summary |
|:-----|:--------|
| `Bfv` | BFV (exact integers). |
| `Ckks` | CKKS (approximate real numbers). |

