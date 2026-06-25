---
title: "SafetyFilter<T>"
description: "Implements comprehensive safety filtering for AI model inputs and outputs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.AdversarialRobustness.Safety`

Implements comprehensive safety filtering for AI model inputs and outputs.

## For Beginners

Think of SafetyFilter as a comprehensive security system
for your AI. It checks everything going in and coming out, looking for anything
suspicious, harmful, or inappropriate. It's like having security guards, content
moderators, and safety inspectors all working together.

## How It Works

SafetyFilter provides multiple layers of protection including input validation,
output filtering, jailbreak detection, and harmful content identification.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SafetyFilter(SafetyFilterOptions<>)` | Initializes a new instance of the safety filter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSafetyScore(Vector<>)` |  |
| `Deserialize(Byte[])` |  |
| `DetectJailbreak(Vector<>)` |  |
| `FilterOutput(Vector<>)` |  |
| `GetOptions` |  |
| `IdentifyHarmfulContent(Vector<>)` |  |
| `InitializePatterns` | Initializes the jailbreak and harmful content detection patterns from options. |
| `LoadModel(String)` |  |
| `Reset` |  |
| `SaveModel(String)` |  |
| `Serialize` |  |
| `ValidateInput(Vector<>)` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `RegexTimeout` | Timeout for regex operations to prevent ReDoS attacks. |

