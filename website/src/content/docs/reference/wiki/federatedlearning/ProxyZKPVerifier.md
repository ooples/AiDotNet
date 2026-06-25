---
title: "ProxyZKPVerifier<T>"
description: "Implements Proxy-based Zero-Knowledge Proof verification for federated learning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.Verification`

Implements Proxy-based Zero-Knowledge Proof verification for federated learning.

## For Beginners

Standard ZK proofs can be expensive for clients to generate.
ProxyZKP uses a semi-trusted proxy (e.g., a TEE enclave or an auditor) that verifies client
computations without seeing the raw data. The client sends commitments to the proxy, the
proxy verifies them using lightweight checks, and issues a cryptographically-signed certificate
to the server. This reduces the ZK proof overhead while maintaining verifiability.

## How It Works

Protocol:

Reference: Proxy-based ZKP for Efficient Federated Verification (2024).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProxyZKPVerifier(Double,Double,Byte[],Int32)` | Creates a new Proxy ZKP verifier. |

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxElementMagnitude` | Gets the maximum allowed element magnitude. |
| `MaxNorm` | Gets the maximum allowed norm. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeCommitment(Dictionary<String,[]>,Byte[])` | Computes an HMAC-SHA256 commitment for a client update. |
| `CryptographicEquals(String,String)` | Constant-time string comparison to prevent timing attacks on hash/signature comparison. |
| `GenerateNonce(Int32)` | Generates a cryptographically secure random nonce. |
| `IssueCertificate(String,Double)` | Issues a signed proxy certificate attesting that the update passed verification. |
| `Verify(Dictionary<String,[]>,String,Byte[])` | Verifies a client update and issues a signed certificate if all checks pass. |
| `VerifyCertificate(ProxyCertificate)` | Verifies a proxy certificate's signature and expiry. |

