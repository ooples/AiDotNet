---
title: "IStreamingEventDetectionSession<T>"
description: "Interface for streaming event detection sessions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for streaming event detection sessions.

## Methods

| Method | Summary |
|:-----|:--------|
| `FeedAudio(Tensor<>)` | Feeds audio samples to the detection session. |
| `GetCurrentState` | Gets current detection state for all event types. |
| `GetNewEvents` | Gets newly detected events since last call. |

## Events

| Event | Summary |
|:-----|:--------|
| `EventDetected` | Event raised when a new event is detected. |

