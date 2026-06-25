---
title: "SafeSerializationBinder"
description: "A custom serialization binder that restricts deserialization to safe types within the AiDotNet namespace."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serialization`

A custom serialization binder that restricts deserialization to safe types within the AiDotNet namespace.

## For Beginners

When loading a saved model from a file, we need to know what types of objects
to create. However, if an attacker crafts a malicious file, they might try to trick the system into
creating dangerous objects. This binder acts as a security guard, only allowing known-safe types.

## How It Works

This binder helps prevent deserialization attacks by only allowing types from the AiDotNet namespace
and common .NET framework types to be deserialized. This is important when using TypeNameHandling.Auto
(or other TypeNameHandling modes) with Newtonsoft.Json.

**Security Note:** Always prefer TypeNameHandling.Auto over TypeNameHandling.All as it minimizes
type information exposure while still supporting polymorphic deserialization.

## Methods

| Method | Summary |
|:-----|:--------|
| `BindToName(Type,String,String)` | Gets the serialized type name for a type during serialization. |
| `BindToType(String,String)` | Gets the type to deserialize given the serialized type name. |
| `IsAllowedType(Type)` | Checks if a resolved type is allowed for deserialization. |

## Fields

| Field | Summary |
|:-----|:--------|
| `AllowedNamespacePrefixes` | Allowed namespace prefixes for deserialization. |
| `AllowedTypeFullNames` | Exact allowed type full names. |
| `_defaultBinder` | The default binder to delegate to for type resolution. |

