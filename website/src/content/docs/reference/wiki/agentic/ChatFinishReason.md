---
title: "ChatFinishReason"
description: "Describes why a chat model stopped generating a response."
section: "API Reference"
---

`Enums` · `AiDotNet.Agentic.Models`

Describes why a chat model stopped generating a response.

## For Beginners

When the model stops "typing", it tells you why. Did it finish its
thought naturally (`Stop`)? Did it run out of room (`Length`)? Did it
pause to ask for a tool to be run (`ToolCalls`)? Was the output blocked by a safety
filter (`ContentFilter`)? This enum captures those outcomes so your code can decide
what to do next.

## How It Works

Every completed (or streamed) response carries a finish reason so callers can react correctly:
for example, retrying with a larger token budget on `Length`, or executing the
requested tools on `ToolCalls`.

## Fields

| Field | Summary |
|:-----|:--------|
| `ContentFilter` | Output was withheld or truncated by a content-safety filter. |
| `Length` | Generation was cut off because the maximum output-token budget was reached. |
| `Stop` | The model finished naturally (it reached a stopping point or a configured stop sequence). |
| `ToolCalls` | The model paused to request one or more tool/function calls. |
| `Unknown` | The provider returned a finish reason that does not map to any of the known values. |

