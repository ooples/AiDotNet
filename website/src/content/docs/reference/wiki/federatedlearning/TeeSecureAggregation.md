---
title: "TeeSecureAggregation<T>"
description: "Performs weighted model aggregation inside a TEE enclave boundary."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

Performs weighted model aggregation inside a TEE enclave boundary.

## For Beginners

This is the core component that makes TEE-based federated learning work.
Instead of aggregating model updates in regular memory (where the server OS can see them),
this class runs aggregation inside a hardware-protected enclave:

## How It Works

**Performance:** TEE aggregation is 10-100x faster than homomorphic encryption because
the enclave operates on plaintext internally. The only overhead is AES encryption/decryption
at the enclave boundary.

**Security model:** The host OS and hypervisor cannot read or modify data inside the
enclave. Even a compromised server cannot extract individual client updates. Only the
aggregated result leaves the enclave.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TeeSecureAggregation(ITeeProvider<>,TeeOptions)` | Initializes a new instance of `TeeSecureAggregation`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AllUpdatesReceived` |  |
| `UpdatesReceived` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate` |  |
| `BeginRound(Int32,Int32)` |  |
| `EncryptForSubmission(Tensor<>)` | Encrypts model parameters for submission to this aggregator. |
| `GenerateSessionKey` |  |
| `GetAttestationQuote(Byte[])` |  |
| `SubmitEncryptedUpdate(Int32,Byte[],Double)` |  |

