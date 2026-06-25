---
title: "RemoteAttestationResult"
description: "Represents the result of verifying a remote attestation quote from a TEE."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

Represents the result of verifying a remote attestation quote from a TEE.

## For Beginners

When a server says "I'm running inside a secure enclave," the client
needs proof. Remote attestation produces a hardware-signed "quote" that a verifier checks against
expected measurements. This class holds the outcome of that verification.

## How It Works

**Key fields:**

## Properties

| Property | Summary |
|:-----|:--------|
| `FailureReason` | Gets or sets a human-readable reason if verification failed. |
| `FirmwareVerified` | Gets or sets whether the platform firmware was verified as up-to-date. |
| `FirmwareVersion` | Gets or sets the platform firmware version string (e.g., TCB level for SGX/TDX). |
| `IsValid` | Gets or sets whether the attestation verification passed all policy checks. |
| `MeasurementHash` | Gets or sets the enclave measurement hash (MRENCLAVE for SGX, launch digest for TDX/SEV-SNP). |
| `PolicyApplied` | Gets or sets the attestation policy that was applied. |
| `ProviderType` | Gets or sets the TEE provider type that produced this attestation. |
| `QuoteTimestamp` | Gets or sets the UTC timestamp when the attestation quote was generated. |
| `RawQuote` | Gets or sets the raw attestation quote bytes (platform-specific binary format). |
| `SignerIdentity` | Gets or sets the enclave signer identity (MRSIGNER for SGX). |

