---
title: "ProxyVerificationResult"
description: "Result of a proxy ZKP verification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Result of a proxy ZKP verification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProxyVerificationResult(Boolean,String,String,Nullable<Double>,ProxyCertificate)` | Creates a new verification result. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Certificate` | Signed proxy certificate, available only if verification passed. |
| `CommitmentHash` | The verified commitment hash, if available. |
| `IsValid` | Whether the verification passed. |
| `Reason` | Human-readable reason for the result. |
| `UpdateNorm` | The L2 norm of the update, if computed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Fail(String)` | Creates a failed verification result. |
| `Pass(String,Double,ProxyCertificate)` | Creates a successful verification result with certificate. |

