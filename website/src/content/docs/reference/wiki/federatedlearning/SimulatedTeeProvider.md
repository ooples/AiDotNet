---
title: "SimulatedTeeProvider<T>"
description: "Software-simulated TEE provider for testing and development without hardware."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

Software-simulated TEE provider for testing and development without hardware.

## For Beginners

Real TEEs require specific hardware (Intel SGX, AMD SEV-SNP, etc.).
This simulated provider lets you develop and test TEE-based federated learning on any machine.
It provides the same API — enclave creation, sealing, attestation — using standard cryptography
instead of hardware isolation.

## How It Works

**Important:** The simulated provider does NOT provide actual hardware security.
It is functionally equivalent (same encrypt/decrypt/attest flow) but offers no protection
against a compromised host. Use `SimulationMode` = false in production.

## Properties

| Property | Summary |
|:-----|:--------|
| `ProviderType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `BuildQuote(Byte[])` |  |
| `ComputeMeasurementHash` |  |
| `DeriveSealingKey` |  |
| `Destroy` |  |
| `GetMaxEnclaveMemory` |  |
| `Initialize(TeeOptions)` |  |

