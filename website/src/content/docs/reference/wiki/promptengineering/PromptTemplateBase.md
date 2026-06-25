---
title: "PromptTemplateBase"
description: "Base class for prompt template implementations providing common functionality and validation."
section: "API Reference"
---

`Base Classes` · `AiDotNet.PromptEngineering.Templates`

Base class for prompt template implementations providing common functionality and validation.

## For Beginners

This is the foundation for all prompt templates.

It handles the common tasks:

- Parsing templates to find variables
- Validating that all required variables are provided
- Substituting variables into the template
- Error checking

When you create a new template type, inherit from this class and you get
all this functionality for free!

## How It Works

This base class handles template parsing, variable extraction, validation, and formatting.
Derived classes can override formatting behavior for specialized template types.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptTemplateBase(String)` | Initializes a new instance of the PromptTemplateBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputVariables` | Gets the list of variable names that this template expects. |
| `Template` | Gets the raw template string before variable substitution. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractVariables(String)` | Extracts variable names from a template string. |
| `Format(Dictionary<String,String>)` | Formats the template with the provided variables to create a complete prompt. |
| `FormatCore(Dictionary<String,String>)` | Core formatting logic to be implemented by derived classes. |
| `Validate(Dictionary<String,String>)` | Validates that the provided variables match the template's requirements. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

