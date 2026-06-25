---
title: "IPromptTemplate"
description: "Defines the contract for prompt templates used in language model interactions."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for prompt templates used in language model interactions.

## For Beginners

A prompt template is like a form with blanks to fill in.

Think of it like a mad lib:

- Template: "The {adjective} {noun} {verb} over the {place}."
- Variables: adjective="quick", noun="fox", verb="jumped", place="fence"
- Result: "The quick fox jumped over the fence."

In LLM applications:

- Template: "Translate the following {source_lang} to {target_lang}: {text}"
- Variables: source_lang="English", target_lang="French", text="Hello"
- Result: "Translate the following English to French: Hello"

Benefits of using templates:

- Reusability: Write the template once, use it many times
- Consistency: All prompts have the same structure
- Maintainability: Change the template in one place
- Safety: Validate inputs before insertion
- Clarity: Separate prompt logic from data

## How It Works

A prompt template provides a structured way to create prompts for language models by combining
a template string with runtime variables. Templates support variable substitution, formatting,
and composition of complex prompts from reusable components.

## Properties

| Property | Summary |
|:-----|:--------|
| `InputVariables` | Gets the list of variable names that this template expects. |
| `Template` | Gets the raw template string before variable substitution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Format(Dictionary<String,String>)` | Formats the template with the provided variables to create a complete prompt. |
| `Validate(Dictionary<String,String>)` | Validates that the provided variables match the template's requirements. |

