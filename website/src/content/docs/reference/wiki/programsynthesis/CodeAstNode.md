---
title: "CodeAstNode"
description: "Represents a node in an abstract syntax tree (AST) for a piece of source code."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.ProgramSynthesis.Models`

Represents a node in an abstract syntax tree (AST) for a piece of source code.

## For Beginners

An AST is a "tree view" of code.

Code is not just text â€” it has structure (functions contain statements, statements contain expressions).
This node represents one item in that tree with a type (Kind) and a location (Span).

## How It Works

This type is intentionally lightweight and is designed for structural inspection and downstream
tasks (understanding, review, search) without requiring consumers to depend on a specific parser.

