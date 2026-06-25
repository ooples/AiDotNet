---
title: "ToolChoiceMode"
description: "Controls whether and how a chat model is allowed to call tools on a given request."
section: "API Reference"
---

`Enums` · `AiDotNet.Agentic.Models`

Controls whether and how a chat model is allowed to call tools on a given request.

## For Beginners

Imagine giving an assistant a toolbox. This setting is your instruction
about the toolbox:

- **Auto**: "Use a tool if you think it helps." (the normal default)
- **None**: "Don't use any tools — just answer with text."
- **Required**: "You must call a tool before answering."

## How It Works

When tools are supplied via `ChatOptions.Tools`, this mode tells the model how aggressively
it may use them. To force a *specific* tool, set `ChatOptions.RequiredToolName` in
addition to selecting `Required`.

## Fields

| Field | Summary |
|:-----|:--------|
| `Auto` | The model decides on its own whether to call a tool or respond directly. |
| `None` | The model is not allowed to call tools; it must respond with content only. |
| `Required` | The model must call at least one tool. |

