---
title: "TrialExpirationReason"
description: "Specifies the reason a trial period expired or a license check failed."
section: "API Reference"
---

`Enums` · `AiDotNet.Exceptions`

Specifies the reason a trial period expired or a license check failed.

## Fields

| Field | Summary |
|:-----|:--------|
| `LicenseExpired` | The provided license key has expired. |
| `LicenseInvalid` | The provided license key is invalid or unrecognized. |
| `OperationLimitReached` | The maximum number of free trial operations (10) has been reached. |
| `SeatLimitReached` | The license has reached its maximum allowed developer seats. |
| `TimeExpired` | The 30-day free trial period has elapsed. |
| `Unknown` | The reason is unknown or not specified. |

