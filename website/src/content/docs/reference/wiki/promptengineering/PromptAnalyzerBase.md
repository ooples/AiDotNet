---
title: "PromptAnalyzerBase"
description: "Provides a base implementation for prompt analyzers with common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.PromptEngineering.Analysis`

Provides a base implementation for prompt analyzers with common functionality.

## For Beginners

This is the foundation that all prompt analyzers build upon.

Think of it like a template for examining prompts:

- It handles common tasks (counting words, finding variables)
- Specific analyzers fill in details like token counting and pattern detection
- This ensures all analyzers work consistently

## How It Works

This abstract class implements the IPromptAnalyzer interface and provides common functionality
for prompt analysis. It handles validation, token counting, and delegates to derived classes
for specific analysis logic.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PromptAnalyzerBase(String,String,Decimal,Func<String,Int32>)` | Initializes a new instance of the PromptAnalyzerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Name` | Gets the name of this analyzer implementation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Analyze(String)` | Analyzes a prompt and returns detailed metrics. |
| `AnalyzeAsync(String,CancellationToken)` | Analyzes a prompt asynchronously. |
| `CalculateComplexity(String,Int32,Int32,Int32)` | Calculates complexity score (0.0 to 1.0). |
| `CalculateCost(Int32)` | Calculates estimated cost based on token count. |
| `CheckForInjection(String)` | Checks for potential prompt injection patterns. |
| `CountExamples(String)` | Counts few-shot examples in the prompt. |
| `CountTokens(String)` | Counts tokens in the given text. |
| `CountVariables(String)` | Counts template variables in the prompt. |
| `CountWords(String)` | Counts words in the text. |
| `CustomValidation(String,ValidationOptions)` | Allows derived classes to add custom validation. |
| `DetectPatterns(String)` | Detects prompt patterns/types. |
| `EnhanceMetrics(String,PromptMetrics)` | Allows derived classes to add custom metrics. |
| `FilterBySeverity(IEnumerable<PromptIssue>,IssueSeverity)` | Filters issues by minimum severity. |
| `ValidatePrompt(String,ValidationOptions)` | Validates a prompt for potential issues. |
| `ValidateVariables(String)` | Validates template variables in the prompt. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

