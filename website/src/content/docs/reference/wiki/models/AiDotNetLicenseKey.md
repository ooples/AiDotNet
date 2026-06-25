---
title: "AiDotNetLicenseKey"
description: "Represents a license key for AiDotNet model encryption and online validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a license key for AiDotNet model encryption and online validation.

## For Beginners

When you purchase a license for AiDotNet, you receive a license key.
This class wraps that key along with optional configuration for connecting to a license server.
You pass it to `AiModelBuilder` so encrypted models can be loaded and saved.

## How It Works

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AiDotNetLicenseKey(String)` | Creates a new `AiDotNetLicenseKey` with the specified key string. |

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableTelemetry` | Gets or sets whether advisory machine-ID telemetry is sent to the license server during validation. |
| `Environment` | Gets or sets the environment label sent during validation (e.g., "production", "staging", "development"). |
| `Key` | Gets the license key string (e.g., "aidn.{id}.{secret}"). |
| `OfflineGracePeriod` | Gets or sets the duration that a cached validation result remains trusted when the server is unreachable. |
| `ServerUrl` | Gets or sets the URL of the license validation server. |

