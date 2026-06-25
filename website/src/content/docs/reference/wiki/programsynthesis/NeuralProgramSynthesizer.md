---
title: "NeuralProgramSynthesizer<T>"
description: "Neural network-based program synthesizer that generates programs from specifications."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Engines`

Neural network-based program synthesizer that generates programs from specifications.

## For Beginners

This AI can write programs for you automatically!

Imagine describing what you want a program to do, or showing examples of
inputs and outputs, and an AI writes the actual code. That's what this does!

You can provide:

- A description: "Write a function that sorts a list of numbers"
- Examples: Input [3,1,2] → Output [1,2,3]
- Or both!

The AI learns from training and generates working code that solves your problem.
It's like having an AI programmer that can code based on your requirements!

## How It Works

NeuralProgramSynthesizer uses deep learning to generate programs from natural language
descriptions, input-output examples, or formal specifications. It employs an encoder-decoder
architecture similar to CodeT5 but optimized for program synthesis tasks.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NeuralProgramSynthesizer(CodeSynthesisArchitecture<>,ICodeModel<>,ILossFunction<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,IProgramExecutionEngine,NeuralProgramSynthesizerOptions,ISqlSyntaxValidator)` | Initializes a new instance of the `NeuralProgramSynthesizer` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `EvaluateProgram(Program<>,ProgramInput<>)` | Evaluates how well a program satisfies the input specification. |
| `GetOptions` |  |
| `RefineProgram(Program<>,ProgramInput<>)` | Refines an existing program to better meet the specification. |
| `SynthesizeProgram(ProgramInput<>)` | Synthesizes a program from the given input specification. |
| `ValidateProgram(Program<>)` | Validates whether a synthesized program is correct and well-formed. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_sqlSyntaxValidator` | Per-instance precise SQL validator; null falls back to the global registration. |

