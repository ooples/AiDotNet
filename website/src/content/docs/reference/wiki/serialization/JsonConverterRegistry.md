---
title: "JsonConverterRegistry"
description: "Registry for JSON converters used in model serialization."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Serialization`

Registry for JSON converters used in model serialization.
Manages custom converters for complex types like Matrix, Vector, and Tensor.

## For Beginners

This class helps convert complex data structures (like matrices and tensors)
into JSON format so they can be saved to files and loaded later. JSON is a text format that's easy to
read and write, making it perfect for saving machine learning models.

## Methods

| Method | Summary |
|:-----|:--------|
| `ClearConverters` | Clears all registered converters. |
| `GetAllConverters` | Gets all registered JSON converters. |
| `GetConvertersForType` | Gets converters that can handle the specified type. |
| `RegisterAllConverters` | Registers all default converters for common types. |
| `RegisterConverter(JsonConverter)` | Registers a custom JSON converter. |

