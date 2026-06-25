---
title: "IStreamingTranscriptionSession<T>"
description: "Interface for streaming transcription sessions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for streaming transcription sessions.

## Methods

| Method | Summary |
|:-----|:--------|
| `FeedAudio(Tensor<>)` | Feeds an audio chunk to the streaming session. |
| `Finalize` | Finalizes the session and returns the complete transcription. |
| `GetPartialResult` | Gets the current partial transcription. |

