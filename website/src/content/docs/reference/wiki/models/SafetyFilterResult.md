---
title: "SafetyFilterResult<T>"
description: "Result of safety filtering on model output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Result of safety filtering on model output.

## Properties

| Property | Summary |
|:-----|:--------|
| `Actions` | Gets or sets detailed filtering actions taken. |
| `DetectedHarmCategories` | Gets or sets the harmful content detected in the output. |
| `FilteredOutput` | Gets or sets the filtered/sanitized output. |
| `FilteringDetails` | Gets or sets additional filtering details. |
| `IsSafe` | Gets or sets whether the output passed filtering. |
| `SafetyScore` | Gets or sets the safety score (0-1, higher is safer). |
| `WasModified` | Gets or sets whether the output was modified during filtering. |

