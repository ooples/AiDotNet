---
title: "ICodeModel<T>"
description: "Represents a code understanding model capable of processing and analyzing source code."
section: "API Reference"
---

`Interfaces` · `AiDotNet.ProgramSynthesis.Interfaces`

Represents a code understanding model capable of processing and analyzing source code.

## For Beginners

A code model is like an AI that understands programming.

Just as language models understand human languages, code models understand programming
languages. They can:

- Read and comprehend code
- Suggest completions while you're writing
- Find bugs and issues
- Explain what code does
- Translate between programming languages

This interface defines what capabilities a code model should have.

## How It Works

ICodeModel defines the interface for models that can understand, encode, and analyze
source code. These models are typically pre-trained on large corpora of code and can
perform tasks like code completion, bug detection, and code summarization.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxSequenceLength` | Gets the maximum sequence length (in tokens) that the model can process. |
| `TargetLanguage` | Gets the target programming language for this model. |
| `VocabularySize` | Gets the vocabulary size of the model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DecodeCode(Tensor<>)` | Decodes a vector representation back into source code. |
| `EncodeCode(String)` | Encodes source code into a vector representation. |
| `GetEmbeddings(String)` | Gets embeddings for code tokens. |
| `PerformTask(CodeTaskRequestBase)` | Performs a code-related task and returns a structured result type. |
| `PerformTask(String,CodeTask)` | Performs a code-related task on the input code. |

