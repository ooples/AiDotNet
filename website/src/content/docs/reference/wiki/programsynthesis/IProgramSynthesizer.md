---
title: "IProgramSynthesizer<T>"
description: "Represents a program synthesis engine capable of automatically generating programs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ProgramSynthesis.Interfaces`

Represents a program synthesis engine capable of automatically generating programs.

## For Beginners

A program synthesizer is like an AI programmer.

Imagine describing what you want a program to do, and an AI writes the code for you.
That's what a program synthesizer does. You provide:

- Examples of inputs and desired outputs
- A description in plain English
- Or formal specifications

And the synthesizer creates a working program that meets your requirements.
This is like having an AI assistant that can code for you!

## How It Works

IProgramSynthesizer defines the interface for models that can automatically generate
programs from specifications, examples, or natural language descriptions. This is a
key component of automated programming and AI-assisted development.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxProgramLength` | Gets the maximum allowed length for synthesized programs. |
| `SynthesisType` | Gets the type of synthesis approach used by this synthesizer. |
| `TargetLanguage` | Gets the target programming language for synthesis. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateProgram(Program<>,ProgramInput<>)` | Evaluates how well a program satisfies the input specification. |
| `RefineProgram(Program<>,ProgramInput<>)` | Refines an existing program to better meet the specification. |
| `SynthesizeProgram(ProgramInput<>)` | Synthesizes a program from the given input specification. |
| `ValidateProgram(Program<>)` | Validates whether a synthesized program is correct and well-formed. |

