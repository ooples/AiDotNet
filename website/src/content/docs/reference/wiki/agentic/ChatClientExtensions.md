---
title: "ChatClientExtensions"
description: "Convenience helpers over `IChatClient` for the common \"prompt in, text out\" case."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Models`

Convenience helpers over `IChatClient` for the common "prompt in, text out" case.

## For Beginners

Instead of constructing a list of messages and reading a response object,
you can write `await client.GenerateTextAsync("Summarize this")` and get a plain string back.

## How It Works

The agentic model layer is message-based, but plenty of callers (and internal consumers such as the
reasoning strategies) just want to send a prompt and read back text. These extensions wrap a single
user message and return the assistant's concatenated text, so simple call sites stay simple while the
full message/tool/streaming API remains available when needed.

## Methods

| Method | Summary |
|:-----|:--------|
| `GenerateResponseAsync(IChatClient<>,String,CancellationToken)` | Sends a single user prompt and returns the assistant's text reply. |
| `GenerateTextAsync(IChatClient<>,String,ChatOptions,CancellationToken)` | Sends a single user prompt and returns the assistant's text reply. |
| `GenerateTextAsync(IChatClient<>,String,String,ChatOptions,CancellationToken)` | Sends a system instruction plus a user prompt and returns the assistant's text reply. |

