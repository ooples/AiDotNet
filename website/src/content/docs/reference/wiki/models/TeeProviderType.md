---
title: "TeeProviderType"
description: "Specifies the Trusted Execution Environment hardware provider."
section: "API Reference"
---

`Enums` · `AiDotNet.Models.Options`

Specifies the Trusted Execution Environment hardware provider.

## For Beginners

A TEE is a secure area in a processor that guarantees code and data
loaded inside are protected from the outside world — even from the operating system.
Different chip vendors provide different TEE implementations:

## Fields

| Field | Summary |
|:-----|:--------|
| `ArmCca` | ARM CCA/Realms — ARM confidential computing architecture. |
| `SevSnp` | AMD SEV-SNP — VM-level memory encryption. |
| `Sgx` | Intel SGX — process-level enclave with 256MB EPC limit. |
| `Simulated` | Software simulation for testing without hardware. |
| `Tdx` | Intel TDX — confidential VM with GB-scale protected memory. |

