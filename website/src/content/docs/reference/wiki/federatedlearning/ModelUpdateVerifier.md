---
title: "ModelUpdateVerifier<T>"
description: "Server-side verification engine that checks all proofs from clients before aggregation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Server-side verification engine that checks all proofs from clients before aggregation.

## For Beginners

The model update verifier is the server-side component that examines
cryptographic proofs from each client before their updates are included in the global model.
Think of it as a "bouncer" that checks credentials before letting clients contribute.

## How It Works

**Verification workflow:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelUpdateVerifier(VerificationOptions)` | Initializes a new instance of `ModelUpdateVerifier`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CommitmentScheme` | Gets the commitment scheme for clients to use. |
| `VerificationHistory` | Gets the verification history for auditing. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetRejectedClientCount(Int32)` | Gets the count of clients that failed verification in a given round. |
| `GetVerifiedClientCount(Int32)` | Gets the count of clients that passed verification in a given round. |
| `VerifyClientUpdate(Int32,Int32,GradientCommitmentData<>,VerificationProof,VerificationProof,VerificationProof)` | Verifies a client's update including commitment and any required proofs. |

