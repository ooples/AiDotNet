---
title: "TensorJsonConverter"
description: "JSON converter for Tensor<T> types."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Serialization`

JSON converter for Tensor<T> types.
Handles serialization and deserialization of tensor objects to/from JSON.

## For Beginners

This class knows how to convert a Tensor (a multi-dimensional array
of numbers) into JSON text format and back. It saves the shape (dimensions) and all the data values,
so the tensor can be perfectly reconstructed later.

## Methods

| Method | Summary |
|:-----|:--------|
| `CanConvert(Type)` | Determines whether this converter can handle the specified type. |
| `ReadJson(JsonReader,Type,Object,JsonSerializer)` | Reads a Tensor<T> object from JSON. |
| `WriteJson(JsonWriter,Object,JsonSerializer)` | Writes a Tensor<T> object to JSON. |

