---
title: "VectorJsonConverter"
description: "JSON converter for Vector<T> types."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serialization`

JSON converter for Vector<T> types.
Handles serialization and deserialization of vector objects to/from JSON.

## For Beginners

This class knows how to convert a Vector (a list of numbers) into
JSON text format and back. It saves the length and all the data values, so the vector can be
perfectly reconstructed later.

## Methods

| Method | Summary |
|:-----|:--------|
| `CanConvert(Type)` | Determines whether this converter can handle the specified type. |
| `ReadJson(JsonReader,Type,Object,JsonSerializer)` | Reads a Vector<T> object from JSON. |
| `WriteJson(JsonWriter,Object,JsonSerializer)` | Writes a Vector<T> object to JSON. |

