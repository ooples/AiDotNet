---
title: "ITeeProvider<T>"
description: "Abstracts a Trusted Execution Environment backend for enclave lifecycle, data sealing, and attestation quote generation."
section: "API Reference"
---

`Interfaces` · `AiDotNet.FederatedLearning.TEE`

Abstracts a Trusted Execution Environment backend for enclave lifecycle, data sealing,
and attestation quote generation.

## For Beginners

A TEE provider manages a hardware-isolated "vault" (enclave) on a processor.
Think of it like a bank's safe-deposit box: you can put data in (seal), take data out (unseal),
and ask the bank for a signed letter proving the box exists (generate attestation quote).

## How It Works

**Implementations:**

## Properties

| Property | Summary |
|:-----|:--------|
| `IsInitialized` | Gets a value indicating whether the enclave is currently initialized and ready. |
| `ProviderType` | Gets the TEE provider type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Destroy` | Destroys the enclave and releases all protected resources. |
| `GenerateAttestationQuote(Byte[])` | Generates a remote attestation quote proving this enclave's identity and integrity. |
| `GetMaxEnclaveMemory` | Gets the maximum memory (in bytes) available inside this enclave. |
| `GetMeasurementHash` | Gets the measurement hash (code identity) of the running enclave. |
| `Initialize(TeeOptions)` | Initializes the TEE enclave. |
| `SealData(Byte[])` | Seals (encrypts) data to the enclave so only this enclave can unseal it. |
| `UnsealData(Byte[])` | Unseals (decrypts) data that was previously sealed by this enclave. |

