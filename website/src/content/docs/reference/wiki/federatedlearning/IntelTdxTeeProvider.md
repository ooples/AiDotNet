---
title: "IntelTdxTeeProvider<T>"
description: "TEE provider for Intel TDX (Trust Domain Extensions) confidential VMs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

TEE provider for Intel TDX (Trust Domain Extensions) confidential VMs.

## For Beginners

Intel TDX is the next generation of Intel's confidential computing,
offering VM-level isolation instead of SGX's process-level enclaves. TDX protects an entire
virtual machine — the hypervisor cannot read or modify the VM's memory. This enables GB-scale
protected workloads, making it ideal for federated learning aggregation with large models.

## How It Works

**TDX vs SGX:**

**Recommended for FL:** TDX is the recommended TEE for federated learning because it
supports large model aggregation without the memory constraints of SGX.

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
| `TdxDefaultMaxBytes` | TDX default max memory: 16 GB (configurable per TD). |

