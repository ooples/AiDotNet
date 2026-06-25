---
title: "Program<T>"
description: "Represents a synthesized program with its source code and metadata."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Models`

Represents a synthesized program with its source code and metadata.

## For Beginners

This class represents a computer program created by AI.

Think of this as a container that holds:

- The actual code (like a recipe holds instructions)
- What language it's written in (Python, Java, etc.)
- Whether the code is valid and will run
- How well it performs
- An optional numerical representation that AI can work with

Just like a recipe card has the recipe, cooking time, and difficulty level,
this class holds a program and all its important information.

## How It Works

The Program class encapsulates a synthesized or analyzed program, including its
source code, the programming language it's written in, validation status, and
optional execution metrics.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Program` | Initializes a new instance of the `Program` class with default values. |
| `Program(String,ProgramLanguage,Boolean,Double,Int32)` | Initializes a new instance of the `Program` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Complexity` | Gets or sets the complexity measure of the program. |
| `Encoding` | Gets or sets the encoded representation of the program. |
| `ErrorMessage` | Gets or sets any error messages from compilation or execution attempts. |
| `ExecutionTimeMs` | Gets or sets execution time in milliseconds if the program was executed. |
| `FitnessScore` | Gets or sets the fitness score of the program. |
| `IsValid` | Gets or sets a value indicating whether the program is syntactically and semantically valid. |
| `Language` | Gets or sets the programming language of the program. |
| `SourceCode` | Gets or sets the source code of the program. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToString` | Returns a string representation of the program. |

