---
title: "PatternDetectionAnalyzer"
description: "Analyzer that specializes in detecting prompt patterns and categorizing prompts."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Analysis`

Analyzer that specializes in detecting prompt patterns and categorizing prompts.

## For Beginners

Figures out what kind of prompt you're using.

Example:

Common patterns detected:

- few-shot: Contains examples
- chain-of-thought: Step-by-step reasoning
- role-playing: Sets up a persona
- template: Contains variables
- question, summarization, translation, etc.

## How It Works

This analyzer identifies what type of prompt is being used and what patterns
are present. It can detect few-shot prompts, chain-of-thought patterns,
system prompts, and various task types.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PatternDetectionAnalyzer` | Initializes a new instance of the PatternDetectionAnalyzer class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CountExamples(String)` | Counts few-shot examples with more sophisticated detection. |
| `DetectOutputFormats(String,List<String>)` | Detects expected output format specifications. |
| `DetectPatterns(String)` | Enhanced pattern detection with more detailed analysis. |
| `DetectStructure(String,List<String>)` | Detects structural patterns in the prompt. |
| `DetectTaskTypes(String,List<String>)` | Detects task types from the prompt. |
| `DetectTechniques(String,String,List<String>)` | Detects prompt engineering techniques. |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Regex timeout to prevent ReDoS attacks. |

