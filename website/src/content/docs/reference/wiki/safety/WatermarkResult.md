---
title: "WatermarkResult"
description: "Detailed result from watermark detection across any modality."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety.Watermarking`

Detailed result from watermark detection across any modality.

## For Beginners

WatermarkResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Detection confidence (0.0 = no watermark, 1.0 = certain). |
| `Modality` | The modality of the content checked (text, image, audio). |
| `WatermarkDetected` | Whether a watermark was detected. |
| `WatermarkType` | The detected watermark type, if identifiable. |

