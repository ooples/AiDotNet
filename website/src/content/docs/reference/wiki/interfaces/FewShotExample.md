---
title: "FewShotExample"
description: "Represents a single few-shot example with input and output."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interfaces`

Represents a single few-shot example with input and output.

## For Beginners

A few-shot example is an input-output pair shown to the model.

Think of it like a flashcard:

- Front (Input): The question or task
- Back (Output): The correct answer or response

Example - Math tutoring:
Input: "What is 5 + 3?"
Output: "5 + 3 = 8"

Example - Code generation:
Input: "Write a function to add two numbers"
Output: "def add(a, b): return a + b"

Example - Translation:
Input: "Good morning"
Output: "Buenos días"

The model learns the pattern from these examples and applies it to new inputs.

## Properties

| Property | Summary |
|:-----|:--------|
| `Input` | Gets or sets the input part of the example. |
| `Metadata` | Gets or sets optional metadata about the example. |
| `Output` | Gets or sets the output part of the example. |

