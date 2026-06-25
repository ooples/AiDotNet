---
title: "YamlLicenseSection"
description: "YAML configuration section for license key settings."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Configuration`

YAML configuration section for license key settings.

## For Beginners

Add a `license` section to your YAML config file to set up
your license key and server URL. You can still override these from code.

## How It Works

**Example YAML:**

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableTelemetry` | Whether to send advisory machine-ID telemetry during validation. |
| `Environment` | The environment label (e.g., "production", "staging", "development"). |
| `Key` | The license key string (e.g., "aidn.{id}.{secret}"). |
| `OfflineGracePeriodDays` | Number of days the cached validation result remains trusted when the server is unreachable. |
| `ServerUrl` | The license validation server URL. |

