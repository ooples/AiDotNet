---
title: "TeeOptions"
description: "Configuration options for Trusted Execution Environment integration in federated learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Trusted Execution Environment integration in federated learning.

## For Beginners

These settings control how the FL server uses hardware security
features. The defaults use simulated mode for testing; switch to a real provider
(Tdx, SevSnp, etc.) for production deployment.

## Properties

| Property | Summary |
|:-----|:--------|
| `Attestation` | Gets or sets the attestation configuration. |
| `ExpectedMeasurement` | Gets or sets the enclave code identity hash (MRENCLAVE for SGX, measurement for others). |
| `MaxEnclaveMemoryMb` | Gets or sets the maximum enclave memory in megabytes. |
| `Policy` | Gets or sets the attestation policy. |
| `Provider` | Gets or sets the TEE provider type. |
| `RequireAttestation` | Gets or sets whether to require remote attestation from clients. |
| `SimulationMode` | Gets or sets whether to enable simulation mode for testing. |

