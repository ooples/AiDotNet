---
title: "ComplexityAnalyzer"
description: "Analyzer that focuses on measuring prompt complexity and structure."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Analysis`

Analyzer that focuses on measuring prompt complexity and structure.

## For Beginners

Measures how complicated your prompt is.

Example:

High complexity might mean:

- Difficult for AI to understand
- Higher chance of misinterpretation
- May need simplification

## How It Works

This analyzer provides detailed complexity metrics including readability scores,
structural analysis, and recommendations for simplification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComplexityAnalyzer(Func<String,Int32>)` | Initializes a new instance of the ComplexityAnalyzer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateNestingDepth(String)` | Calculates maximum nesting depth in the prompt. |
| `CalculateReadability(String)` | Calculates Flesch Reading Ease score (0-100, higher = easier). |
| `CountInstructions(String)` | Counts instruction-related keywords. |
| `CountSentences(String)` | Counts sentences in the text. |
| `CountSyllables(String)` | Estimates syllable count in text. |
| `CountWordSyllables(String)` | Estimates syllable count for a single word. |
| `CustomValidation(String,ValidationOptions)` | Adds custom validation for complexity-related issues. |
| `EnhanceMetrics(String,PromptMetrics)` | Adds detailed complexity metrics to the analysis. |
| `IsVowel(Char)` | Checks if a character is a vowel. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

