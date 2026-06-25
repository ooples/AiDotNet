---
title: "IChatPromptTemplate"
description: "Renders a chat conversation (a list of `ChatMessage`) into the single prompt string a local language model is fed."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Agentic.Models.Local`

Renders a chat conversation (a list of `ChatMessage`) into the single prompt string a local
language model is fed. Different model families expect different role markers, so this is pluggable.

## For Beginners

Cloud chat APIs accept a list of messages directly, but a raw local model
just continues one block of text. This converts the conversation into that block, tagging who said what
(system/user/assistant) in the format the model was trained on, and ends with the assistant's turn so the
model knows it should reply next.

## Methods

| Method | Summary |
|:-----|:--------|
| `Render(IReadOnlyList<ChatMessage>)` | Renders the conversation into a prompt string. |

