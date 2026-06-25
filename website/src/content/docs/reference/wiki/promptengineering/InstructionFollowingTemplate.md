---
title: "InstructionFollowingTemplate"
description: "Template optimized for clear, structured instruction-following tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

Template optimized for clear, structured instruction-following tasks.

## For Beginners

Structures instructions for clear AI execution.

Example:

Benefits:

- Clear separation of objective, steps, and constraints
- Numbered instructions for sequential execution
- Explicit constraints prevent unwanted behavior
- Input/output sections clearly marked

## How It Works

This template provides a clear structure for instructions, context, and constraints,
helping models understand and follow complex multi-step instructions accurately.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `InstructionFollowingTemplate` | Initializes a new instance of the InstructionFollowingTemplate class. |
| `InstructionFollowingTemplate(String)` | Initializes a new instance with a custom template string. |
| `InstructionFollowingTemplate(String,List<String>,List<String>,String,String)` | Initializes a new instance with specified components. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddConstraint(String)` | Adds a constraint. |
| `AddConstraints(String[])` | Adds multiple constraints. |
| `AddInstruction(String)` | Adds an instruction step. |
| `AddInstructions(String[])` | Adds multiple instruction steps. |
| `Builder` | Creates a builder for constructing instruction-following templates. |
| `Classification(String[])` | Creates a classification template. |
| `FormatCore(Dictionary<String,String>)` | Formats the instruction-following template. |
| `QuestionAnswering` | Creates a question-answering template. |
| `Summarization(Int32)` | Creates a summarization template. |
| `Translation(String)` | Creates a translation template. |
| `WithInputDescription(String)` | Sets the input description. |
| `WithObjective(String)` | Sets the objective for the task. |
| `WithOutputDescription(String)` | Sets the output description. |

