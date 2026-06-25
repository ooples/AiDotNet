---
title: "JsonSchemaGenerator"
description: "Generates JSON Schema (the dialect chat models consume for tool parameters and structured output) from .NET types and method parameters using reflection."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Tools`

Generates JSON Schema (the dialect chat models consume for tool parameters and structured output)
from .NET types and method parameters using reflection.

## For Beginners

A model needs a description of a tool's inputs written in "JSON Schema".
Writing that by hand is tedious and error-prone. This class reads your C# types and writes the schema
for you — e.g. a method taking `(string city, int days = 3)` becomes an object with a required
string `city` and an optional integer `days`.

## How It Works

This is what lets a plain C# method become a model-callable tool without hand-writing a schema.
Primitive types map to their JSON counterparts, enums become string enumerations, collections become
arrays, string-keyed dictionaries become open objects, and other classes become nested objects built
from their public readable properties (with a depth/cycle guard so recursive types terminate).

## Methods

| Method | Summary |
|:-----|:--------|
| `ForParameters(IReadOnlyList<ParameterInfo>)` | Builds an object schema (`type: object` with `properties` and `required`) describing a method's parameters. |
| `ForType(Type)` | Builds a JSON Schema fragment describing a single .NET type. |

