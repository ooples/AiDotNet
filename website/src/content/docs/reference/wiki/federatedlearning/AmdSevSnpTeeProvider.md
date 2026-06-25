---
title: "AmdSevSnpTeeProvider<T>"
description: "TEE provider for AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

TEE provider for AMD SEV-SNP (Secure Encrypted Virtualization - Secure Nested Paging).

## For Beginners

AMD SEV-SNP encrypts all of a virtual machine's memory using a
hardware key that the hypervisor cannot access. "SNP" adds integrity protection — if anyone
tries to tamper with encrypted memory, the CPU detects it. This makes it possible to run
federated learning aggregation in a VM that even the cloud provider's host OS cannot inspect.

## How It Works

**Key concepts:**

**Cloud availability:** Azure DCasv5/ECasv5 VMs, Google Cloud C2D, AWS (forthcoming).

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

## Fields

| Field | Summary |
|:-----|:--------|
| `SevSnpDefaultMaxBytes` | SEV-SNP supports full VM memory, default 32 GB for FL workloads. |

