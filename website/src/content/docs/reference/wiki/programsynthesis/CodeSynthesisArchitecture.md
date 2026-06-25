---
title: "CodeSynthesisArchitecture<T>"
description: "Defines the architecture configuration for code synthesis and understanding models."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Models`

Defines the architecture configuration for code synthesis and understanding models.

## For Beginners

This is a blueprint for building AI models that understand code.

Just like TransformerArchitecture defines how to build a general transformer,
CodeSynthesisArchitecture defines how to build models specifically for:

- Understanding code
- Generating code
- Translating between programming languages
- Finding bugs
- Completing code

It includes all the settings needed to build these specialized code models,
like which programming language to work with and how much code it can handle.

## How It Works

CodeSynthesisArchitecture extends the neural network architecture with code-specific
parameters such as programming language, maximum code length, vocabulary size, and
synthesis strategy. It serves as a blueprint for building code models like CodeBERT,
GraphCodeBERT, and CodeT5.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodeSynthesisArchitecture(SynthesisType,ProgramLanguage,CodeTask,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Int32,Double,Boolean,Boolean,NetworkComplexity,Int32,Int32,List<ILayer<>>)` | Initializes a new instance of the `CodeSynthesisArchitecture` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `CodeTaskType` | Gets the code task type this architecture is optimized for. |
| `DropoutRate` | Gets the dropout rate for regularization. |
| `FeedForwardDimension` | Gets the feed-forward network dimension. |
| `MaxProgramLength` | Gets the maximum allowed program length for synthesis. |
| `MaxSequenceLength` | Gets the maximum sequence length (in tokens). |
| `ModelDimension` | Gets the model dimension (embedding size). |
| `NumDecoderLayers` | Gets the number of decoder layers (for generation tasks). |
| `NumEncoderLayers` | Gets the number of encoder layers. |
| `NumHeads` | Gets the number of attention heads. |
| `SynthesisType` | Gets the type of synthesis approach to use. |
| `TargetLanguage` | Gets the target programming language. |
| `UseDataFlow` | Gets whether to use data flow information (for GraphCodeBERT-style models). |
| `UsePositionalEncoding` | Gets whether to use positional encoding. |
| `VocabularySize` | Gets the vocabulary size. |

