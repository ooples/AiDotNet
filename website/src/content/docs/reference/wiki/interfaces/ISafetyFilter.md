---
title: "ISafetyFilter<T>"
description: "Defines the contract for safety filters that detect and prevent harmful or inappropriate model inputs and outputs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for safety filters that detect and prevent harmful or inappropriate model inputs and outputs.

## How It Works

Safety filters act as gatekeepers that monitor model inputs and outputs to prevent
harmful, inappropriate, or malicious content from passing through the system.

**For Beginners:** Think of safety filters as "security guards" for your AI system.
They check everything going in and coming out to make sure nothing dangerous or
inappropriate gets through.

Common safety filter functions include:

- Input Validation: Check that inputs are safe and properly formatted
- Output Filtering: Ensure outputs don't contain harmful content
- Jailbreak Detection: Identify attempts to bypass safety measures
- Harmful Content Detection: Flag potentially dangerous or inappropriate content

Why safety filters matter:

- They prevent misuse of AI systems
- They protect users from harmful content
- They help maintain ethical AI deployments
- They catch edge cases and adversarial inputs

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeSafetyScore(Vector<>)` | Computes a safety score for model inputs or outputs. |
| `DetectJailbreak(Vector<>)` | Detects jailbreak attempts that try to bypass safety measures. |
| `FilterOutput(Vector<>)` | Filters model outputs to remove or flag harmful content. |
| `GetOptions` | Gets the configuration options for the safety filter. |
| `IdentifyHarmfulContent(Vector<>)` | Identifies harmful or inappropriate content in text or data. |
| `Reset` | Resets the safety filter state. |
| `ValidateInput(Vector<>)` | Validates that an input is safe and appropriate for processing. |

