---
title: "SafetyReport"
description: "Unified safety report aggregating findings from all safety modules in the pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Safety`

Unified safety report aggregating findings from all safety modules in the pipeline.

## For Beginners

After the safety system finishes checking your content,
it produces this report summarizing everything it found. The key properties are:

- `IsSafe`: Quick yes/no answer
- `OverallAction`: What the system recommends doing
- `Findings`: Detailed list of every issue found

## How It Works

A SafetyReport is the single output object from the safety pipeline. It aggregates
all findings from every safety module that ran, provides an overall safety verdict,
and recommends the strictest action needed.

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectedCategories` | Gets the categories of harmful content detected. |
| `EvaluationTimeMs` | Gets the duration of the safety evaluation in milliseconds. |
| `Findings` | Gets the list of all safety findings from all modules. |
| `HighestSeverity` | Gets the highest severity found across all findings. |
| `IsSafe` | Gets whether the content passed all safety checks. |
| `ModulesExecuted` | Gets the names of all safety modules that were executed. |
| `OverallAction` | Gets the overall recommended action (the strictest action from all findings). |
| `OverallScore` | Gets the overall safety score (0.0 = completely unsafe, 1.0 = completely safe). |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromFindings(IReadOnlyList<SafetyFinding>,IReadOnlyList<String>,Double)` | Creates a report from a list of findings. |
| `Safe(IReadOnlyList<String>,Double)` | Creates a safe (no findings) report. |

