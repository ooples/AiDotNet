---
title: "TeeAttestationOptions"
description: "Configuration options for TEE remote attestation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for TEE remote attestation.

## For Beginners

Remote attestation lets you verify that a remote computer is truly
running inside a secure enclave. These options control how that verification works —
what evidence to expect and how long it's valid.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExpectedMeasurements` | Gets or sets the expected enclave measurement hashes. |
| `ExpectedSignerIdentity` | Gets or sets the expected signer identity hash (hex-encoded). |
| `MaxQuoteAgeSec` | Gets or sets the maximum age of an attestation quote in seconds before it's considered stale. |
| `QuoteFormat` | Gets or sets the attestation quote format. |
| `VerifyPlatformFirmware` | Gets or sets whether to verify the platform firmware is up to date. |
| `VerifySignerIdentity` | Gets or sets whether to check the enclave signer identity (MRSIGNER for SGX). |

