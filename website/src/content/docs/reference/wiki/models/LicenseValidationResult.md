---
title: "LicenseValidationResult"
description: "Contains the result of a license key validation attempt."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Contains the result of a license key validation attempt.

## For Beginners

After the license validator contacts the server (or checks its cache),
this object tells you whether the key is valid, what tier the user is on, how many seats are used,
and when the key expires.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LicenseValidationResult(LicenseKeyStatus,String,Nullable<DateTimeOffset>,Int32,Nullable<Int32>,Nullable<DateTimeOffset>,String,Byte[],IReadOnlyList<String>)` | Creates a new `LicenseValidationResult`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Capabilities` | Gets the namespace-prefixed capability strings (e.g., `tensors:save`, `tensors:load`, `model:save`, `model:load`) the server returned for the validated license. |
| `DecryptionToken` | Gets a copy of the server-side decryption token for Layer 2 key escrow, or null if not available. |
| `ExpiresAt` | Gets the expiration date of the license, or null if it does not expire. |
| `Message` | Gets an optional human-readable message from the server. |
| `SeatsMax` | Gets the maximum number of seats allowed for this license, or null if unlimited. |
| `SeatsUsed` | Gets the number of seats currently in use for this license. |
| `Status` | Gets the validation status of the license key. |
| `Tier` | Gets the subscription tier associated with this license, or null if unknown. |
| `ValidatedAt` | Gets the UTC timestamp of when this validation was performed. |

## Methods

| Method | Summary |
|:-----|:--------|
| `HasCapability(String)` | Returns true if the server granted the named capability for this license. |

