---
title: "PromptMetrics"
description: "Contains metrics and analysis results for a prompt."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Analysis`

Contains metrics and analysis results for a prompt.

## For Beginners

This is a report card for your prompt.

When you analyze a prompt, you get back this object with all the measurements:

- Token count: How many "words" the AI sees (affects cost)
- Estimated cost: How much this prompt will cost in API fees
- Complexity score: How complicated the prompt is (0-1)
- Variable count: How many {placeholders} are in the prompt
- Detected patterns: What type of prompt this is (question, instruction, etc.)

Example usage:

## How It Works

This class encapsulates all the measurements and analysis data produced when
analyzing a prompt. It includes token counts, cost estimates, complexity scores,
and detected patterns that help developers understand and optimize their prompts.

## Properties

| Property | Summary |
|:-----|:--------|
| `AnalyzedAt` | Gets or sets the timestamp when this analysis was performed. |
| `CharacterCount` | Gets or sets the character count of the prompt. |
| `ComplexityScore` | Gets or sets the complexity score of the prompt (0.0 to 1.0). |
| `DetectedPatterns` | Gets or sets the detected prompt patterns or types. |
| `EstimatedCost` | Gets or sets the estimated API cost for this prompt. |
| `ExampleCount` | Gets or sets the count of examples included in the prompt (for few-shot prompts). |
| `ModelName` | Gets or sets the name of the model used for token counting. |
| `TokenCount` | Gets or sets the total token count of the prompt. |
| `VariableCount` | Gets or sets the number of template variables in the prompt. |
| `WordCount` | Gets or sets the word count of the prompt. |

