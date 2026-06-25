---
title: "AttestationPolicy"
description: "Specifies the attestation policy for TEE remote attestation verification."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the attestation policy for TEE remote attestation verification.

## For Beginners

Remote attestation lets a client verify that the server is actually
running inside a genuine TEE (not just pretending). The policy controls how strict this
verification is:

## Fields

| Field | Summary |
|:-----|:--------|
| `Custom` | User-defined verification logic. |
| `Relaxed` | Allows minor version differences in enclave measurements. |
| `Strict` | Exact measurement match required. |

