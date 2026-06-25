---
title: "CodeT5<T>"
description: "CodeT5 is an encoder-decoder model for code understanding and generation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Engines`

CodeT5 is an encoder-decoder model for code understanding and generation.

## For Beginners

CodeT5 can both understand AND generate code.

Unlike CodeBERT which mainly understands code, CodeT5 can also create it:

- Understand: Read and analyze code (encoder)
- Generate: Write new code (decoder)

This makes it powerful for tasks like:

- Translating Python to Java
- Generating code from English descriptions
- Creating documentation from code
- Fixing bugs by rewriting code

Think of it as both a reader and a writer, not just a reader.

## How It Works

CodeT5 is based on the T5 (Text-To-Text Transfer Transformer) architecture adapted
for code. It uses an encoder-decoder structure that can handle both code understanding
and generation tasks. It's particularly effective for code translation, summarization,
and generation from natural language descriptions.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodeT5(CodeSynthesisArchitecture<>,ILossFunction<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ITokenizer)` | Initializes a new instance of the `CodeT5` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `NumDecoderLayers` | Gets the number of decoder layers. |
| `NumEncoderLayers` | Gets the number of encoder layers. |

