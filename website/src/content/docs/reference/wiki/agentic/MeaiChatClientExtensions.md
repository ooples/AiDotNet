---
title: "MeaiChatClientExtensions"
description: "Fluent bridges between AiDotNet's `IChatClient` and Microsoft.Extensions.AI's `IChatClient`, in both directions, with full tool-calling support."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Models.Connectors`

Fluent bridges between AiDotNet's `IChatClient` and Microsoft.Extensions.AI's
`IChatClient`, in both directions, with full tool-calling support.

## Methods

| Method | Summary |
|:-----|:--------|
| `AsAgenticChatClient(IChatClient,String)` | Wraps a Microsoft.Extensions.AI chat client as an AiDotNet `IChatClient`, inheriting the .NET ecosystem's connectors (OpenAI, Azure, Ollama, …) — including tool calling. |
| `AsMeaiChatClient(IChatClient<>)` | Exposes an AiDotNet chat client as a Microsoft.Extensions.AI client, so MEAI-aware code (Semantic Kernel, the MEAI middleware pipeline, etc.) can consume an AiDotNet model — including its tool calls. |

