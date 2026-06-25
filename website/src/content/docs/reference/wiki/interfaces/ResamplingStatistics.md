---
title: "ResamplingStatistics<T>"
description: "Contains statistics about a resampling operation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Contains statistics about a resampling operation.

## For Beginners

This class tells you what the resampling strategy did to your data,
so you can verify it worked as expected.

## Properties

| Property | Summary |
|:-----|:--------|
| `OriginalClassCounts` | The original number of samples per class. |
| `ResampledClassCounts` | The new number of samples per class after resampling. |
| `SamplesAddedPerClass` | The number of samples added per class (positive for oversampling). |
| `SamplesRemovedPerClass` | The number of samples removed per class (positive for undersampling). |
| `TotalOriginalSamples` | Total samples before resampling. |
| `TotalResampledSamples` | Total samples after resampling. |

