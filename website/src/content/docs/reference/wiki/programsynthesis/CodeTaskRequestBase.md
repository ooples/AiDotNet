---
title: "CodeTaskRequestBase"
description: "Base type for all code task execution requests."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ProgramSynthesis.Requests`

Base type for all code task execution requests.

## For Beginners

This is the common "envelope" for any code task request.

Think of this as a form you submit to ask the system to do something with code.
The specific task (like Search or CodeReview) determines which extra fields you need to fill in.

## How It Works

This request model is used as the canonical input shape for all `CodeTask` operations.
Concrete request types (e.g., completion, search, code review) add task-specific fields.

## Properties

| Property | Summary |
|:-----|:--------|
| `Language` | Gets or sets the primary language context for the request. |
| `MaxWallClockMilliseconds` | Gets or sets an optional wall-clock time budget (in milliseconds) for the request. |
| `RequestId` | Gets or sets an optional request identifier for correlation and tracing. |
| `SqlDialect` | Gets or sets the SQL dialect to use when `Language` is `SQL`. |
| `Task` | Gets the requested task. |

