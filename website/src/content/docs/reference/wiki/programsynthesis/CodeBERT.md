---
title: "CodeBERT<T>"
description: "CodeBERT is a bimodal pre-trained model for programming and natural languages."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ProgramSynthesis.Engines`

CodeBERT is a bimodal pre-trained model for programming and natural languages.

## For Beginners

CodeBERT is an AI that understands programming languages.

Just like BERT understands English, CodeBERT understands code. It's been trained
on millions of code examples from GitHub and can:

- Understand what code does
- Find similar code
- Complete code as you write
- Generate documentation
- Translate between code and descriptions

Think of it as an AI that's read millions of lines of code and learned the
patterns of good programming, just like you learn language by reading books.

## How It Works

CodeBERT is designed to understand both code and natural language. It uses a
transformer-based encoder architecture pre-trained on code-documentation pairs
from GitHub. It excels at tasks like code search, code documentation generation,
and code completion.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CodeBERT(CodeSynthesisArchitecture<>,ILossFunction<>,IGradientBasedOptimizer<,Tensor<>,Tensor<>>,ITokenizer)` | Initializes a new instance of the `CodeBERT` class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeLayers` | Initializes the layers of the CodeBERT model. |

