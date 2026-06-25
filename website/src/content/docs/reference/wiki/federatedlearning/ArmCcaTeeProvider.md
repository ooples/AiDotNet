---
title: "ArmCcaTeeProvider<T>"
description: "TEE provider for ARM CCA (Confidential Compute Architecture) / Realm Management Extension."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

TEE provider for ARM CCA (Confidential Compute Architecture) / Realm Management Extension.

## For Beginners

ARM CCA introduces "Realms" — isolated execution environments
on ARM processors (Armv9+). A Realm is like an enclave that runs a full OS or application,
protected from the hypervisor and other Realms. ARM CCA is important for edge/IoT federated
learning because many edge devices use ARM processors.

## How It Works

**Key concepts:**

**Target hardware:** Armv9-A with RME (Realm Management Extension), available in
ARM Neoverse V2+ and Cortex-X4+ cores.

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
| `CcaDefaultMaxBytes` | ARM CCA default max Realm memory: 8 GB for edge/cloud workloads. |

