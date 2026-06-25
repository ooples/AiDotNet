---
title: "ByteTrack<T>"
description: "ByteTrack: Multi-Object Tracking by Associating Every Detection Box."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Tracking`

ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

## For Beginners

ByteTrack improves upon SORT by using almost all detection
boxes (including low-confidence ones) for tracking. It uses a two-stage association:
first matching high-confidence detections, then low-confidence ones.

## How It Works

Key features:

- Associates all detection boxes (high and low confidence)
- Two-stage matching strategy
- Recovers occluded objects using low-score detections
- State-of-the-art performance on MOT benchmarks

Reference: Zhang et al., "ByteTrack: Multi-Object Tracking by Associating
Every Detection Box", ECCV 2022

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ByteTrack(TrackingOptions<>)` | Creates a new ByteTrack tracker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` |  |
| `Update(List<Detection<>>)` |  |

