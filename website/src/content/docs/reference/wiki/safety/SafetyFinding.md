---
title: "SafetyFinding"
description: "Represents a single safety finding from a safety module evaluation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety`

Represents a single safety finding from a safety module evaluation.

## For Beginners

When the safety system checks your content, it may find
zero or more problems. Each problem is represented as a SafetyFinding with details
about what was detected and how serious it is.

## How It Works

A safety finding captures one specific issue detected during safety analysis,
including what was found, how severe it is, where it was found, and what action
is recommended.

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Gets the safety category of this finding. |
| `Confidence` | Gets the confidence score for this finding (0.0 to 1.0). |
| `Description` | Gets a human-readable description of the finding. |
| `Excerpt` | Gets the excerpt of content that triggered this finding. |
| `RecommendedAction` | Gets the recommended action for this finding. |
| `Severity` | Gets the severity level of this finding. |
| `SourceModule` | Gets the name of the safety module that produced this finding. |
| `SpanEnd` | Gets the end offset (exclusive) of the finding in the input. |
| `SpanStart` | Gets the start offset (character index or sample index) of the finding in the input. |

