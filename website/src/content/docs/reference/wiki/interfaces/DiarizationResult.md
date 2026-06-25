---
title: "DiarizationResult<T>"
description: "Result of speaker diarization."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of speaker diarization.

## Properties

| Property | Summary |
|:-----|:--------|
| `Duration` | Gets the audio duration in seconds (alias for TotalDuration for legacy API compatibility). |
| `NumSpeakers` | Gets or sets the number of unique speakers detected. |
| `OverlapRegions` | Gets or sets overlapping speech regions (if detected). |
| `Segments` | Gets or sets the detected speaker segments. |
| `SpeakerLabels` | Gets or sets the unique speaker labels. |
| `SpeakerStats` | Gets speaker statistics (speaking time, number of turns). |
| `TotalDuration` | Gets or sets the total audio duration in seconds. |
| `Turns` | Gets the speaker turns (alias for Segments for legacy API compatibility). |

