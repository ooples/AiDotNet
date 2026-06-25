---
title: "DeepSORT<T>"
description: "DeepSORT (Deep SORT) tracking with appearance features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Tracking`

DeepSORT (Deep SORT) tracking with appearance features.

## For Beginners

DeepSORT extends SORT by adding deep appearance features
(Re-ID embeddings) to improve association accuracy, especially for occlusions and
ID switches. It uses a cascade matching strategy.

## How It Works

Key features:

- Deep appearance descriptor (Re-ID network)
- Cascade matching (prioritize recent tracks)
- Combined motion and appearance matching
- Gallery of appearance features per track

Reference: Wojke et al., "Simple Online and Realtime Tracking with a Deep
Association Metric", ICIP 2017

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DeepSORT(TrackingOptions<>)` | Creates a new DeepSORT tracker. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` |  |
| `Update(List<Detection<>>)` |  |
| `Update(List<Detection<>>,Tensor<>)` |  |

