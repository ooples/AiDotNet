---
title: "SerializationExtensions"
description: "Provides extension methods for serializing and deserializing data used in AI models."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Extensions`

Provides extension methods for serializing and deserializing data used in AI models.

## For Beginners

Serialization is the process of converting data structures or objects into a format 
that can be stored (in a file or database) or transmitted (across a network). Deserialization is the reverse process.

## How It Works

This class provides methods to save your AI model data to files and load them back later.
Think of it like saving your progress in a video game so you can continue later.

## Methods

| Method | Summary |
|:-----|:--------|
| `ReadArray(BinaryReader)` | Reads an array of type T from a binary stream. |
| `ReadInt32Array(BinaryReader)` | Reads an array of integers from a binary stream. |
| `ReadValue(BinaryReader,Type)` | Reads a value of the specified type from a binary stream. |
| `WriteArray(BinaryWriter,[])` | Writes an array of type T to a binary stream. |
| `WriteInt32Array(BinaryWriter,Int32[])` | Writes an array of integers to a binary stream. |
| `WriteValue(BinaryWriter,Object)` | Writes a value of a supported type to a binary stream. |

