---
title: "IRemoteAttestationVerifier"
description: "Verifies remote attestation quotes from TEE enclaves."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.TEE`

Verifies remote attestation quotes from TEE enclaves.

## For Beginners

When a client receives a claim "I'm running in a secure enclave,"
it needs to verify that claim. The verifier checks the hardware-signed attestation quote
against expected measurements, signer identity, and freshness policies.

## How It Works

**Verification steps:**

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedProvider` | Gets the supported TEE provider type for this verifier. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Verify(Byte[],Byte[],TeeAttestationOptions,AttestationPolicy)` | Verifies a remote attestation quote against the given policy and options. |

