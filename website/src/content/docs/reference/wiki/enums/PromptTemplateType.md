---
title: "PromptTemplateType"
description: "Represents different types of prompt templates for language model interactions."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Represents different types of prompt templates for language model interactions.

## For Beginners

Prompt templates are pre-structured formats for communicating with language models.

Think of templates like email templates:

- Instead of writing each email from scratch, you use a template with placeholders
- You fill in the specific details (name, date, etc.) for each email
- The overall structure and tone remain consistent

Prompt templates work the same way:

- You create a template with placeholders for variable content
- You fill in specific values when you need to use the template
- The language model receives a well-structured, consistent prompt

Different template types serve different purposes, from simple variable substitution to
complex multi-turn conversations with examples and tool usage.

## Fields

| Field | Summary |
|:-----|:--------|
| `ChainOfThought` | Template for chain-of-thought reasoning with step-by-step problem solving. |
| `Chat` | Template for structured message-based conversations with roles (system, user, assistant). |
| `FewShot` | Template with few-shot learning examples to guide the model's output. |
| `Optimized` | Template optimized through automated prompt engineering techniques. |
| `ReAct` | Template for ReAct (Reasoning + Acting) pattern combining thought and action. |
| `Simple` | Simple template with variable substitution using placeholders. |
| `Tool` | Template that includes function/tool calling capabilities. |

