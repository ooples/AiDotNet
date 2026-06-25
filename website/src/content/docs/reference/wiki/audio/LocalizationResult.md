---
title: "LocalizationResult"
description: "Result of sound source localization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Localization`

Result of sound source localization.

## For Beginners

LocalizationResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Algorithm` | Algorithm used for localization. |
| `AzimuthDegrees` | Estimated azimuth angle in degrees (-180 to 180). |
| `Confidence` | Confidence of the estimate (0-1). |
| `ElevationDegrees` | Estimated elevation angle in degrees (-90 to 90). |
| `TdoaSamples` | Time difference of arrival in samples. |
| `TdoaSeconds` | Time difference of arrival in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetDirectionVector` | Gets direction as unit vector (x, y, z). |

