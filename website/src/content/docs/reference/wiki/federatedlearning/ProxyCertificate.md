---
title: "ProxyCertificate"
description: "A signed certificate issued by the proxy after verifying a client update."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

A signed certificate issued by the proxy after verifying a client update.

## For Beginners

This is like a stamp of approval from the proxy. It contains
what was verified (the commitment hash and norm), when (timestamps), and a cryptographic
signature that proves the proxy issued it. The server can check this signature to trust the
update without re-running all checks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProxyCertificate(String,Double,Int64,Int64,String)` | Creates a new proxy certificate. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CommitmentHash` | The commitment hash that was verified. |
| `ExpiresAtTicks` | When the certificate expires (UTC ticks). |
| `IsExpired` | Whether the certificate has expired. |
| `IssuedAtTicks` | When the certificate was issued (UTC ticks). |
| `Signature` | HMAC-SHA256 signature over the certificate contents. |
| `UpdateNorm` | The L2 norm of the verified update. |

