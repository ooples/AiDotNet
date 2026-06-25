---
title: "PromptValidator"
description: "Specialized prompt validator with comprehensive validation rules."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Analysis`

Specialized prompt validator with comprehensive validation rules.

## For Beginners

Checks your prompt for problems before you use it.

Example:

What it checks:

- Syntax errors (missing braces, unclosed quotes)
- Security issues (potential injection attacks)
- Best practices (length, clarity)
- Model compatibility

## How It Works

This validator performs detailed validation of prompts, checking for
common issues, security concerns, and best practice violations.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptValidator(ValidationOptions,IPromptAnalyzer)` | Initializes a new instance of the PromptValidator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetSummary(String,ValidationOptions)` | Gets a quick summary of validation results. |
| `Validate(String,ValidationOptions)` | Validates a prompt and returns all detected issues. |
| `ValidateBasic(String)` | Validates basic prompt requirements. |
| `ValidateBestPractices(String)` | Validates best practices. |
| `ValidateLength(String,ValidationOptions)` | Validates prompt length. |
| `ValidateSecurity(String)` | Validates prompt security. |
| `ValidateSyntax(String)` | Validates prompt syntax. |
| `ValidateVariables(String)` | Validates template variables. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

