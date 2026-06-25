---
title: "LossThresholdProof<T>"
description: "Proves that a client's local training loss is below a threshold without revealing the actual value."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Proves that a client's local training loss is below a threshold without revealing the actual value.

## For Beginners

After local training, each client has a training loss value. If the
loss is too high, the client's model may be poorly trained or even poisoned. This proof lets
the server reject clients with high loss without seeing the actual loss value (which could
leak information about the client's private dataset).

## How It Works

**How it works:**

**Reference:** ZKP-FedEval (2025) — privacy-preserving FL evaluation using ZKPs.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LossThresholdProof(IZkProofSystem,Double)` | Initializes a new instance of `LossThresholdProof`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `ProofSystemName` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateLossProof(Double,Int32,Int32)` | Generates a proof that the given loss value is below the threshold. |
| `GenerateProof(Byte[],VerificationConstraint)` |  |
| `Verify(VerificationProof,VerificationConstraint)` |  |

