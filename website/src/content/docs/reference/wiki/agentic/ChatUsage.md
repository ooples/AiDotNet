---
title: "ChatUsage"
description: "Token accounting for a chat request: how many tokens went in and came out."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

Token accounting for a chat request: how many tokens went in and came out.

## For Beginners

Models bill by "tokens" (roughly word-pieces). This records how many
tokens your prompt used and how many the reply used, so you can measure and control spend.

## How It Works

Usage drives cost tracking and budgeting. Providers report input (prompt) and output (completion)
token counts; `TotalTokens` is their sum.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ChatUsage(Int32,Int32)` | Initializes a new `ChatUsage`. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputTokens` | Gets the number of input (prompt) tokens. |
| `OutputTokens` | Gets the number of output (completion) tokens. |
| `TotalTokens` | Gets the total number of tokens (`InputTokens` + `OutputTokens`). |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` |  |

