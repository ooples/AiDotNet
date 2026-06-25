---
title: "ChatRequestContext"
description: "The mutable request state flowing through a chat-middleware pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Pipeline`

The mutable request state flowing through a chat-middleware pipeline. Middleware can inspect and rewrite
the `Messages` and `Options` before the model is called, and share state via
`Items`.

## For Beginners

Think of this as the request envelope passed down an assembly line. Each
station (middleware) can read it, change it (add a system instruction, tweak settings), or stash a note in
the bag for later stations, before it reaches the model.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatRequestContext(IReadOnlyList<ChatMessage>,ChatOptions)` | Initializes a new request context. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Items` | Gets a property bag for sharing state between middleware stages. |
| `Messages` | Gets or sets the conversation to send (middleware may rewrite it, but never to `null`). |
| `Options` | Gets or sets the per-call options (middleware may rewrite them). |

