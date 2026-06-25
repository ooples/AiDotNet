---
title: "QualityLevel"
description: "Quality levels for adaptive inference on resource-constrained devices."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Quality levels for adaptive inference on resource-constrained devices.

## How It Works

**For Beginners:** When running AI models on devices with limited resources (like phones or
edge devices), you often need to trade off quality for speed or battery life. This enum lets you
choose how the model should balance these tradeoffs:

- **Low**: Fastest inference, uses least battery, but lower accuracy. Good for when battery is low

or the device is under heavy load.

- **Medium**: Balanced - decent speed and accuracy. Good default for most situations.
- **High**: Best accuracy, slower inference, uses more battery. Use when you need the best results

and the device has plenty of power.

The library can automatically switch between these levels based on battery level and CPU load.

## Fields

| Field | Summary |
|:-----|:--------|
| `High` | High quality, maximum accuracy - prioritizes accuracy over speed. |
| `Low` | Low quality, maximum speed - prioritizes performance over accuracy. |
| `Medium` | Medium quality, balanced - good compromise between speed and accuracy. |

