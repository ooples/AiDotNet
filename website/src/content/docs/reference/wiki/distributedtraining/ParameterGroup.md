---
title: "ParameterGroup<T>"
description: "Represents a group of parameters that should be communicated together."
section: "API Reference"
---

`Models & Types` · `AiDotNet.DistributedTraining`

Represents a group of parameters that should be communicated together.

## For Beginners

This is like a shipping box that contains multiple items. Each ParameterGroup
represents a chunk of parameters that will be sent together in one communication.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsMerged` | Indicates whether this group was created by merging smaller groups. |
| `Name` | A descriptive name for this parameter group (e.g., "Layer1.Weights"). |
| `Size` | The number of parameters in this group. |
| `StartIndex` | The starting index of this group in the full parameter vector. |

