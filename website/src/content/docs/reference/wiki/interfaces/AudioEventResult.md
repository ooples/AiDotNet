---
title: "AudioEventResult<T>"
description: "Result of audio event detection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Result of audio event detection.

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectedEventTypes` | Gets or sets the unique event types detected. |
| `EventStats` | Gets or sets event statistics. |
| `Events` | Gets or sets the detected events. |
| `TotalDuration` | Gets or sets the total duration in seconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetEnumerator` | Returns an enumerator that iterates through the events. |
| `GetEventsByType(String)` | Gets events of a specific type. |
| `System#Collections#IEnumerable#GetEnumerator` | Returns an enumerator that iterates through the events. |

