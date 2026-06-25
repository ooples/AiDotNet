---
title: "RolePlayingTemplate"
description: "Template that creates persona-based prompts for role-playing scenarios."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

Template that creates persona-based prompts for role-playing scenarios.

## For Beginners

Makes the AI adopt a specific character or expert role.

Example:

Benefits:

- Consistent expertise level in responses
- Domain-specific language and knowledge
- Appropriate communication style
- Better task-relevant responses

## How It Works

This template helps models adopt specific personas, expertise levels,
and communication styles for more targeted and consistent responses.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `RolePlayingTemplate(String)` | Initializes a new instance with a custom template string. |
| `RolePlayingTemplate(String,IEnumerable<String>,String,String)` | Initializes a new instance of the RolePlayingTemplate class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Builder` | Creates a builder for constructing role-playing templates. |
| `BusinessAnalyst(String)` | Creates a business analyst persona. |
| `CodeReviewer(String[])` | Creates a code reviewer persona. |
| `CreativeWriter(String)` | Creates a creative writer persona. |
| `FormatCore(Dictionary<String,String>)` | Formats the role-playing template. |
| `Teacher(String,String)` | Creates a teacher/educator persona. |
| `TechnicalExpert(String,String)` | Creates a technical expert persona. |
| `WithTask(String)` | Sets the task for the persona to perform. |

