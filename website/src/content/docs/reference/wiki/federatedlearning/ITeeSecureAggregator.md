---
title: "ITeeSecureAggregator<T>"
description: "Performs model aggregation inside a TEE enclave boundary."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.TEE`

Performs model aggregation inside a TEE enclave boundary.

## For Beginners

In standard federated learning, the server aggregates client model updates
in plaintext — the server can see every client's update. TEE-based aggregation runs inside a secure
enclave: clients encrypt their updates so only the enclave can decrypt them, the enclave aggregates
in plaintext internally, and returns the sealed result. The host OS never sees individual updates.

## How It Works

**Flow:**

**Performance:** 10-100x faster than homomorphic encryption because aggregation runs in
plaintext inside the enclave — the overhead is only encryption/decryption at ingress/egress.

## Properties

| Property | Summary |
|:-----|:--------|
| `AllUpdatesReceived` | Gets a value indicating whether all expected clients have submitted their updates. |
| `UpdatesReceived` | Gets the number of updates received so far this round. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Aggregate` | Performs weighted aggregation of all submitted updates inside the TEE enclave. |
| `BeginRound(Int32,Int32)` | Initializes the aggregator for a new round. |
| `GenerateSessionKey` | Generates a session key that clients use to encrypt their updates for this round. |
| `GetAttestationQuote(Byte[])` | Gets the attestation quote for this aggregation enclave so clients can verify it. |
| `SubmitEncryptedUpdate(Int32,Byte[],Double)` | Submits an encrypted client model update to the enclave for aggregation. |

