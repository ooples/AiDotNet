---
title: "AnchorMatcher<T>"
description: "Matches anchor boxes to ground truth boxes for training object detectors."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Anchors`

Matches anchor boxes to ground truth boxes for training object detectors.

## For Beginners

During training, we need to assign each anchor box to either:

- A ground truth box (positive sample): The anchor should predict this object
- Background (negative sample): The anchor should predict "no object"
- Ignore: The anchor is borderline and excluded from loss calculation

This matching process determines which anchors learn to detect which objects.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AnchorMatcher` | Creates a new anchor matcher with default thresholds. |
| `AnchorMatcher(Double,Double,Boolean)` | Creates a new anchor matcher with custom thresholds. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AllowLowQualityMatches` | Whether to allow low-quality matches (best anchor for each GT even if below threshold). |
| `NegativeThreshold` | IoU threshold below which an anchor is considered negative. |
| `PositiveThreshold` | IoU threshold above which an anchor is considered positive. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Match(List<BoundingBox<>>,List<BoundingBox<>>)` | Matches anchors to ground truth boxes. |
| `MatchCenterBased(List<ValueTuple<Double,Double>>,List<BoundingBox<>>,List<Int32>)` | Matches anchors using center-based assignment (used in FCOS, YOLOX). |
| `MatchSimOTA(List<BoundingBox<>>,List<BoundingBox<>>,List<BoundingBox<>>,List<Double>)` | Matches anchors using SimOTA (used in YOLOX). |

