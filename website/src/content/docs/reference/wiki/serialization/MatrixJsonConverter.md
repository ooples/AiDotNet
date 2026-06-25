---
title: "MatrixJsonConverter"
description: "JSON converter for Matrix<T> types."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serialization`

JSON converter for Matrix<T> types.
Handles serialization and deserialization of matrix objects to/from JSON.

## For Beginners

This class knows how to convert a Matrix (a grid of numbers) into
JSON text format and back. It saves the number of rows, columns, and all the data values,
so the matrix can be perfectly reconstructed later.

## Methods

| Method | Summary |
|:-----|:--------|
| `CanConvert(Type)` | Determines whether this converter can handle the specified type. |
| `ReadJson(JsonReader,Type,Object,JsonSerializer)` | Reads a Matrix<T> object from JSON. |
| `WriteJson(JsonWriter,Object,JsonSerializer)` | Writes a Matrix<T> object to JSON. |

