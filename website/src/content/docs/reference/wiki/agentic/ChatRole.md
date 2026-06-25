---
title: "ChatRole"
description: "Identifies who authored a `ChatMessage` in a conversation."
section: "API Reference"
---

`Enums` · `AiDotNet.Agentic.Models`

Identifies who authored a `ChatMessage` in a conversation.

## For Beginners

Think of a chat as a transcript of a conversation. Every line in the
transcript has a speaker. This enum is the list of possible speakers:

- **System**: the setup instructions ("You are a helpful assistant that answers in French").
- **User**: the human's messages.
- **Assistant**: the model's own replies.
- **Tool**: the result of running a tool/function the model asked to call.

Using a fixed set of roles (an enum) instead of free-text strings means a typo like "asistant"
can't slip through and break a request at runtime.

## How It Works

Modern chat models are message-based rather than prompt-based: instead of one big string,
a request is a list of messages, each tagged with the role of its author. The role tells the
model how to treat the content (instructions, user input, its own prior replies, or tool output).

## Fields

| Field | Summary |
|:-----|:--------|
| `Assistant` | Output authored by the model itself. |
| `System` | High-level instructions that steer the model's behavior for the whole conversation. |
| `Tool` | The result produced by executing a tool/function that the assistant requested. |
| `User` | Input authored by the end user (the human asking questions or giving instructions). |

