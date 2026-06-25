---
title: "CopyrightDetectorConfig"
description: "Configuration for copyright and memorization detection modules."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Safety.Text`

Configuration for copyright and memorization detection modules.

## For Beginners

Use this to configure copyright detection. You can set
the n-gram size for overlap detection and the threshold for flagging content
as potentially copyrighted or memorized.

## Properties

| Property | Summary |
|:-----|:--------|
| `MinTextLength` | Minimum text length to analyze. |
| `NgramSize` | N-gram size for overlap detection. |
| `Threshold` | Memorization score threshold (0.0-1.0). |

