---
title: "IntelSgxTeeProvider<T>"
description: "TEE provider for Intel SGX (Software Guard Extensions) process-level enclaves."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FederatedLearning.TEE`

TEE provider for Intel SGX (Software Guard Extensions) process-level enclaves.

## For Beginners

Intel SGX creates small, secure "enclaves" inside a regular process.
Data inside an enclave is encrypted in memory and invisible to the OS, hypervisor, or other
processes. SGX enclaves are limited to ~256 MB of protected memory (EPC), making them suitable
for aggregation of model updates but not for training full models.

## How It Works

**Key SGX concepts:**

**Production use:** Requires an SGX-capable CPU (Xeon Scalable 3rd gen+) and the Intel
SGX SDK/PSW installed. This class models the SGX enclave lifecycle; actual SGX calls would be
made via P/Invoke to the SGX SDK in a production deployment.

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
| `SgxEpcLimitBytes` | SGX Enclave Page Cache limit: 256 MB. |

