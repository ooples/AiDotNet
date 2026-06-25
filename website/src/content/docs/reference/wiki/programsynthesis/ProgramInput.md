---
title: "ProgramInput<T>"
description: "Represents the input specification for program synthesis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Models`

Represents the input specification for program synthesis.

## For Beginners

This class describes what you want the program to do.

When you want AI to create a program for you, you need to tell it what you want.
This class lets you provide that information in different ways:

- Describe it in plain English
- Give examples of inputs and expected outputs
- Specify constraints (like "must run in under 1 second")

Think of it like ordering at a restaurant - you tell the chef what you want,
and they create the dish. This is how you tell the AI what program you want.

## How It Works

ProgramInput encapsulates all the information needed to synthesize a program,
including natural language descriptions, input-output examples, formal specifications,
and constraints.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ProgramInput` | Initializes a new instance of the `ProgramInput` class with default values. |
| `ProgramInput(String,ProgramLanguage,List<ProgramInputOutputExample>,List<String>)` | Initializes a new instance of the `ProgramInput` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Constraints` | Gets or sets constraints that the synthesized program must satisfy. |
| `Description` | Gets or sets the natural language description of the desired program. |
| `Encoding` | Gets or sets an encoded representation of the input for neural processing. |
| `Examples` | Gets or sets the input-output examples for inductive synthesis. |
| `FormalSpecification` | Gets or sets the formal specification in logic or a domain-specific language. |
| `MaxComplexity` | Gets or sets the maximum allowed complexity for the synthesized program. |
| `Tags` | Gets or sets metadata tags for categorizing or filtering synthesis tasks. |
| `TargetLanguage` | Gets or sets the target programming language for synthesis. |
| `TestCases` | Gets or sets the test cases for program validation. |
| `TimeoutMs` | Gets or sets the timeout for program synthesis in milliseconds. |

## Methods

| Method | Summary |
|:-----|:--------|
| `AddConstraint(String)` | Adds a constraint to the Constraints list. |
| `AddExample(String,String)` | Adds an input-output example to the Examples list. |
| `AddTestCase(String,String)` | Adds a test case to the TestCases list. |

