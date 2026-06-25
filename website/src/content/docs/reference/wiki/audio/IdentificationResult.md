---
title: "IdentificationResult"
description: "Result of speaker identification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.Speaker`

Result of speaker identification.

## For Beginners

IdentificationResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `IdentifiedSpeakerId` | Gets or sets the identified speaker ID (null if no match above threshold). |
| `Matches` | Gets or sets ranked matches for all enrolled speakers. |
| `Threshold` | Gets or sets the threshold used for identification. |
| `TopScore` | Gets or sets the top match score. |

