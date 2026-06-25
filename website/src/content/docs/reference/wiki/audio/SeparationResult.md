---
title: "SeparationResult<T>"
description: "Result of music source separation containing individual stems."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Audio.SourceSeparation`

Result of music source separation containing individual stems.

## For Beginners

SeparationResult provides AI safety functionality. Default values follow the original paper settings.

## Properties

| Property | Summary |
|:-----|:--------|
| `Bass` | Isolated bass track. |
| `Drums` | Isolated drums/percussion track. |
| `Other` | Other instruments (guitar, piano, etc.). |
| `SampleRate` | Sample rate of output stems. |
| `Vocals` | Isolated vocal track. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToDictionary` | Gets all stems as a dictionary. |

