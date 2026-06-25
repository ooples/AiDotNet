---
title: "AiContent"
description: "Base type for a single piece of content inside a `ChatMessage`."
section: "API Reference"
---

`Base Classes` · `AiDotNet.Agentic.Models`

Base type for a single piece of content inside a `ChatMessage`.

## For Beginners

Older APIs treated a message as one block of text. Real conversations
are richer — a single turn might contain a sentence *and* an image, or the model might
reply with "please run the calculator tool" instead of text. Representing a message as a list of
typed parts lets us model all of that cleanly. Each part is one of the subclasses of this class.

## How It Works

A chat message is not just a string — it is a list of content parts. This lets one message mix
modalities and structured items: plain text, images, a request from the model to call a tool
(`ToolCallContent`), or the result of running a tool (`ToolResultContent`).
Concrete subclasses are matched with pattern matching, e.g. `part is TextContent t`.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AiContent` | Initializes the base content part. |

