---
title: "ClassifierRegistry<T>"
description: "Registry for creating classifier instances by type name, enabling serialization and deserialization of wrapped classifiers in Meta and MultiLabel classifiers."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Classification`

Registry for creating classifier instances by type name, enabling serialization
and deserialization of wrapped classifiers in Meta and MultiLabel classifiers.

## For Beginners

When a Meta or MultiLabel classifier wraps other classifiers
(e.g., BaggingClassifier wraps an array of base classifiers), we need a way to
save and restore those wrapped classifiers. The registry maps classifier type names
to concrete types that can be instantiated for deserialization.

## How It Works

All built-in AiDotNet classifiers are automatically registered. If you create custom
classifiers, register them with `Register` before deserializing
models that wrap them.

**Usage:**

## Properties

| Property | Summary |
|:-----|:--------|
| `RegisteredTypes` | Returns all registered type names. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(String)` | Creates a new, untrained classifier instance by type name. |
| `DeserializeClassifier(String,String)` | Deserializes a wrapped classifier from a type name and base64-encoded data. |
| `EnsureInitialized` | Ensures all built-in classifiers are registered. |
| `GetTypeName(IClassifier<>)` | Gets the short type name for a classifier instance. |
| `IsRegistered(String)` | Checks whether a type name is registered. |
| `Register` | Registers a classifier type using generics. |
| `Register(String,Type)` | Registers a classifier type by name and Type object. |
| `SerializeClassifier(IClassifier<>)` | Serializes a wrapped classifier into a type name and base64-encoded data pair. |

