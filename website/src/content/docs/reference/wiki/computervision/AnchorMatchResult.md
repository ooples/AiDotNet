---
title: "AnchorMatchResult<T>"
description: "Result of anchor-to-ground-truth matching."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ComputerVision.Detection.Anchors`

Result of anchor-to-ground-truth matching.

## Properties

| Property | Summary |
|:-----|:--------|
| `Labels` | Label for each anchor (positive, negative, or ignore). |
| `MatchedGtIndices` | For each anchor, the index of the matched GT box (-1 if no match). |
| `MatchedIoUs` | IoU between each anchor and its matched GT box. |
| `NumIgnored` | Gets the number of ignored anchors. |
| `NumNegative` | Gets the number of negative matches. |
| `NumPositive` | Gets the number of positive matches. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetNegativeIndices` | Gets indices of all negative anchors. |
| `GetPositiveIndices` | Gets indices of all positive anchors. |

