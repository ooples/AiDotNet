---
title: "KeyDetectionResult"
description: "Result of key detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.MusicAnalysis`

Result of key detection.

## For Beginners

KeyDetectionResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Confidence` | Gets or sets the confidence score (0-1). |
| `Correlation` | Gets or sets the Pearson correlation with the key profile (-1 to 1). |
| `KeyIndex` | Gets or sets the key index (0 = C, 1 = C#, etc.). |
| `Mode` | Gets or sets the key mode (major or minor). |
| `Name` | Gets or sets the full key name (e.g., "C major", "A minor"). |
| `RelativeKey` | Gets or sets the relative major/minor key. |
| `RootNote` | Gets or sets the root note name (e.g., "C", "A"). |

