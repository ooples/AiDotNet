---
title: "ChatMessage"
description: "A single message in a chat conversation: a `ChatRole` plus one or more content parts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

A single message in a chat conversation: a `ChatRole` plus one or more content parts.

## For Beginners

This is one line in the conversation transcript. The most common case is
a bit of text from the user or the assistant, so there are shortcuts for that:
`ChatMessage.User("Hello")`, `ChatMessage.System("You are helpful")`,
`ChatMessage.Assistant("Hi!")`. For tool calling there are richer parts, but the shortcuts
cover everyday use.

## How It Works

A chat request is an ordered list of `ChatMessage` values. Each message is authored by
a role (system/user/assistant/tool) and carries a list of `AiContent` parts so it can
mix text, images, tool-call requests, and tool results. Messages are immutable once constructed.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatMessage(ChatRole,IReadOnlyList<AiContent>,String)` | Initializes a new message from a role and an explicit list of content parts. |
| `ChatMessage(ChatRole,String,String)` | Initializes a new message from a role and a single piece of text. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AuthorName` | Gets the optional author name associated with this message. |
| `Contents` | Gets the ordered, immutable list of content parts that make up this message. |
| `Role` | Gets the role that authored this message. |
| `Text` | Gets the concatenated text of all `TextContent` parts in this message. |
| `ToolCalls` | Gets the tool-call requests contained in this message (typically present on assistant messages whose finish reason was `ToolCalls`). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Assistant(IReadOnlyList<AiContent>)` | Creates an assistant message from explicit content parts (for example, one or more `ToolCallContent` parts the model produced). |
| `Assistant(String)` | Creates an assistant message from text. |
| `System(String)` | Creates a system message carrying high-level instructions. |
| `Tool(String,String,Boolean)` | Creates a tool-result message answering a prior `ToolCallContent`. |
| `User(String)` | Creates a user message. |

