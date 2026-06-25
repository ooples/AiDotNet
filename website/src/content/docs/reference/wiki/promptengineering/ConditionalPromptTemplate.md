---
title: "ConditionalPromptTemplate"
description: "Template that supports conditional sections based on variable presence or values."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

Template that supports conditional sections based on variable presence or values.

## For Beginners

Include parts of the prompt only when conditions are met.

Example:

## How It Works

This template allows including or excluding sections based on conditions,
making prompts more dynamic and context-aware.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ConditionalPromptTemplate(String)` | Initializes a new instance of the ConditionalPromptTemplate class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FormatCore(Dictionary<String,String>)` | Formats the template, evaluating all conditional sections. |
| `Validate(Dictionary<String,String>)` | Validates variables - only requires non-conditional variables. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

