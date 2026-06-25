---
title: "PromptIssue"
description: "Represents an issue or warning detected during prompt validation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Analysis`

Represents an issue or warning detected during prompt validation.

## For Beginners

A problem found in your prompt.

Example issues:

- Warning: "Prompt length (5000 tokens) approaches limit (8192)"
- Error: "Unclosed variable placeholder at position 45"
- Info: "Consider adding examples for better results"

## How It Works

When validating a prompt, various issues may be detected. This class
represents a single issue with its severity, message, and location.

## Properties

| Property | Summary |
|:-----|:--------|
| `Code` | Gets or sets the issue code for programmatic handling. |
| `Length` | Gets or sets the length of the problematic text (if applicable). |
| `Message` | Gets or sets the human-readable message describing the issue. |
| `Position` | Gets or sets the character position where the issue was detected (if applicable). |
| `Severity` | Gets or sets the severity level of the issue. |

