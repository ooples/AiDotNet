---
title: "VerificationOptions"
description: "Configuration options for zero-knowledge verification in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for zero-knowledge verification in federated learning.

## For Beginners

These settings control how client updates are verified before
aggregation. The default uses hash commitments with norm bound checking, which provides
a good balance of security and performance.

## Properties

| Property | Summary |
|:-----|:--------|
| `Commitment` | Gets or sets the commitment options for the underlying commitment scheme. |
| `ElementBound` | Gets or sets the per-element bound for ElementBound verification. |
| `GradientNormBound` | Gets or sets the gradient L2 norm bound for NormBound verification. |
| `Level` | Gets or sets the verification level. |
| `LossThreshold` | Gets or sets the loss threshold for LossThreshold verification. |
| `ProofSystem` | Gets or sets the ZK proof system to use. |
| `ProofTimeoutMs` | Gets or sets the maximum time allowed for proof generation per client, in milliseconds. |
| `RejectFailedClients` | Gets or sets whether to reject clients that fail verification. |
| `SecurityParameterBits` | Gets or sets the security parameter in bits. |

