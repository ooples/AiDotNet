---
title: "TeeProviderBase<T>"
description: "Base class for TEE providers with common enclave lifecycle, sealing, and attestation logic."
section: "API Reference"
---

`Base Classes` · `AiDotNet.FederatedLearning.TEE`

Base class for TEE providers with common enclave lifecycle, sealing, and attestation logic.

## For Beginners

All TEE providers share common patterns — initializing an enclave,
sealing/unsealing data, generating attestation quotes. This base class implements those patterns
so that each hardware-specific provider (SGX, TDX, SEV-SNP, etc.) only needs to implement
the platform-specific parts.

## How It Works

**Data sealing** uses AES-256-GCM with a key derived from the enclave identity.
In simulation mode the key is derived from HKDF; in production the hardware provides a
sealing key bound to the enclave measurement.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInitialized` |  |
| `Options` | Gets the current TEE options. |
| `ProviderType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildQuote(Byte[])` | Builds a platform-specific attestation quote. |
| `ComputeMeasurementHash` | Computes the measurement hash (code identity) of this enclave. |
| `DeriveSealingKey` | Derives the sealing key for this enclave. |
| `Destroy` |  |
| `EnsureInitialized` | Throws if the enclave has not been initialized. |
| `GenerateAttestationQuote(Byte[])` |  |
| `GetMaxEnclaveMemory` |  |
| `GetMeasurementHash` |  |
| `Initialize(TeeOptions)` |  |
| `SealData(Byte[])` |  |
| `UnsealData(Byte[])` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `MinSealedDataLength` | Minimum sealed payload size in bytes: AES-GCM nonce (12) + tag (16) + at least 1 byte of ciphertext. |

