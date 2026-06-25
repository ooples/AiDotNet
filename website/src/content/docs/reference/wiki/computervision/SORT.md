---
title: "SORT<T>"
description: "SORT (Simple Online and Realtime Tracking) implementation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Tracking`

SORT (Simple Online and Realtime Tracking) implementation.

## For Beginners

SORT is a simple yet effective online tracking algorithm
that uses Kalman filtering for state estimation and the Hungarian algorithm for
detection-to-track association based on IoU overlap.

## How It Works

Key features:

- Kalman filter for motion prediction
- IoU-based association using Hungarian algorithm
- No appearance features (pure motion-based)
- Real-time performance (~260 FPS)

Reference: Bewley et al., "Simple Online and Realtime Tracking", ICIP 2016

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SORT(TrackingOptions<>)` | Creates a new SORT tracker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` |  |
| `Update(List<Detection<>>)` |  |

