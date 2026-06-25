---
title: "IStreamingSynthesisSession<T>"
description: "Interface for streaming TTS synthesis sessions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Interface for streaming TTS synthesis sessions.

## Methods

| Method | Summary |
|:-----|:--------|
| `FeedText(String)` | Feeds text to the streaming session. |
| `Finalize` | Finalizes the session and returns any remaining audio. |
| `GetAvailableAudio` | Gets available audio chunks that have been synthesized. |

