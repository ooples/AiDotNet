---
title: "LicenseKeyStatus"
description: "Represents the current status of a license key after validation."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents the current status of a license key after validation.

## Fields

| Field | Summary |
|:-----|:--------|
| `Active` | The license key is valid and active. |
| `Expired` | The license key has expired. |
| `Invalid` | The license key is not valid (wrong format, unknown key, or incorrect). |
| `Revoked` | The license key has been revoked by an administrator. |
| `SeatLimitReached` | The license key has reached its maximum number of allowed seats. |
| `ValidationPending` | The license key could not be validated online, but the client is allowed to continue operating until the offline grace period expires. |

