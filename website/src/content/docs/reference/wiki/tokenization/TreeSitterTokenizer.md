---
title: "TreeSitterTokenizer"
description: "AST-aware tokenizer using Tree-sitter for parsing source code into syntax trees."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Tokenization.CodeTokenization`

AST-aware tokenizer using Tree-sitter for parsing source code into syntax trees.
Provides structure-aware tokenization that understands programming language grammar.

## For Beginners

Think of this tokenizer as a code-reading expert that truly
understands programming languages. While simple tokenizers just split text on spaces and
punctuation (like cutting a sentence into individual words), Tree-sitter actually reads
and understands the code's structure.

For example, when parsing "function add(a, b) { return a + b; }":

- A simple tokenizer sees: ["function", "add", "(", "a", ",", "b", ")", "{", ...]
- Tree-sitter sees: A function declaration named "add" with parameters "a" and "b",

containing a return statement with a binary expression.

This deeper understanding helps machine learning models learn code patterns more effectively,
because tokens are grouped by their semantic role (function names, variable names, operators, etc.)
rather than just their text content.

## How It Works

Tree-sitter is an incremental parsing library that builds concrete syntax trees for source code.
Unlike simple regex-based tokenizers, Tree-sitter understands the actual structure of code,
enabling more intelligent tokenization that preserves semantic meaning.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TreeSitterTokenizer(ITokenizer,TreeSitterLanguage,Boolean,Boolean)` | Creates a new Tree-sitter tokenizer for the specified programming language. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CleanupTokens(List<String>)` | Cleans up tokens and converts them back to text. |
| `CreateCSharp(ITokenizer,Boolean)` | Creates a Tree-sitter tokenizer for C# code. |
| `CreateJava(ITokenizer,Boolean)` | Creates a Tree-sitter tokenizer for Java code. |
| `CreateJavaScript(ITokenizer,Boolean)` | Creates a Tree-sitter tokenizer for JavaScript code. |
| `CreatePython(ITokenizer,Boolean)` | Creates a Tree-sitter tokenizer for Python code. |
| `Dispose` | Releases the resources used by the Tree-sitter parser. |
| `Dispose(Boolean)` | Releases the resources used by the Tree-sitter parser. |
| `ExtractTokensWithQueries(Node,List<String>)` | Extracts tokens from the AST using Tree-sitter queries. |
| `Finalize` | Finalizer to ensure resources are released. |
| `GetQueryPattern` | Gets the Tree-sitter query pattern for the current language. |
| `GetTreeSitterLanguageSpec(TreeSitterLanguage)` | Gets the Tree-sitter language library/function spec for a language enum value. |
| `NormalizeNodeType(String)` | Normalizes an AST node type to a consistent format for token prefixes. |
| `Tokenize(String)` | Tokenizes source code using AST-aware parsing. |

