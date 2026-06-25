---
title: "PartitionStrategy"
description: "Strategies for partitioning models between cloud and edge devices."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Strategies for partitioning models between cloud and edge devices.

## How It Works

**For Beginners:** Sometimes you want to split an AI model so that part of it runs on a
local device (edge) and part runs in the cloud. This is useful for:

- Reducing bandwidth by processing some data locally
- Improving privacy by keeping sensitive data on the device
- Balancing speed (edge processing) with power (cloud processing)

Different strategies determine where to split the model:

- **EarlyLayers**: First few layers run on edge, rest in cloud. Good for preprocessing data

locally before sending it to the cloud.

- **LateLayers**: Most processing on edge, only final layers in cloud. Good for devices with

decent processing power.

- **Balanced**: Split in the middle. Good general-purpose strategy.
- **Adaptive**: Automatically determines the best split based on network speed, device power,

and battery level.

- **Manual**: You specify exactly where to split. For advanced users.

## Fields

| Field | Summary |
|:-----|:--------|
| `Adaptive` | Adaptively determine partition based on runtime conditions (network speed, device power, battery level, model size). |
| `Balanced` | Balanced partition - splits model in the middle. |
| `EarlyLayers` | Execute early layers on edge, rest on cloud. |
| `LateLayers` | Execute most layers on edge, only final on cloud. |
| `Manual` | Manual partition specification - you control exactly where to split. |

