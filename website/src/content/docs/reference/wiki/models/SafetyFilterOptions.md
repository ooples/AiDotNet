---
title: "SafetyFilterOptions<T>"
description: "Configuration options for safety filtering mechanisms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for safety filtering mechanisms.

## For Beginners

These settings control how strict your "security guards" are.
You can adjust sensitivity thresholds, what types of content to filter, and how thoroughly
to check for problems.

## How It Works

These options control how inputs and outputs are validated and filtered to prevent
harmful or inappropriate content from passing through the AI system.

## Properties

| Property | Summary |
|:-----|:--------|
| `EnableInputValidation` | Gets or sets whether to enable input validation. |
| `EnableOutputFiltering` | Gets or sets whether to enable output filtering. |
| `HarmfulContentCategories` | Gets or sets the harmful content categories to check for. |
| `JailbreakSensitivity` | Gets or sets the jailbreak detection sensitivity. |
| `LogFilePath` | Gets or sets the file path used when logging filtered content. |
| `LogFilteredContent` | Gets or sets whether to log filtered content for review. |
| `MaxInputLength` | Gets or sets the maximum input length to process. |
| `SafetyThreshold` | Gets or sets the safety threshold for content filtering. |
| `UseClassifier` | Gets or sets whether to use a classifier for harmful content detection. |

