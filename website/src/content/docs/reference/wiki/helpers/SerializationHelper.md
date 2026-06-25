---
title: "SerializationHelper<T>"
description: "Provides methods for serializing and deserializing AI model components to and from binary formats."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides methods for serializing and deserializing AI model components to and from binary formats.

## How It Works

**For Beginners:** Serialization is the process of converting complex data structures (like AI models) 
into a format that can be easily stored or transmitted. Think of it like saving your game progress 
so you can continue later. This helper makes it possible to save trained AI models to disk and 
load them back when needed.

## Methods

| Method | Summary |
|:-----|:--------|
| `DeserializeMatrix(BinaryReader)` | Deserializes a matrix from a binary reader without specifying expected dimensions. |
| `DeserializeMatrix(BinaryReader,Int32,Int32)` | Deserializes a matrix from a binary format with expected dimensions. |
| `DeserializeMatrix(Byte[])` | Deserializes a matrix from a byte array. |
| `DeserializeNode(BinaryReader)` | Deserializes a decision tree node and its children from a binary format. |
| `DeserializeTensor(BinaryReader)` | Deserializes a tensor from a binary stream. |
| `DeserializeVector(BinaryReader)` | Deserializes a vector from a binary reader without specifying expected length. |
| `DeserializeVector(BinaryReader,Int32)` | Deserializes a vector from a binary format with an expected length. |
| `DeserializeVector(Byte[])` | Deserializes a vector from a byte array. |
| `ReadValue(BinaryReader)` | Reads a value of type T from a binary reader. |
| `SerializeInterface(BinaryWriter,)` | Serializes an interface instance by writing its type name to a BinaryWriter. |
| `SerializeMatrix(BinaryWriter,Matrix<>)` | Serializes a matrix to a binary format. |
| `SerializeMatrix(Matrix<>)` | Serializes a matrix to a byte array. |
| `SerializeNode(DecisionTreeNode<>,BinaryWriter)` | Serializes a decision tree node and its children to a binary format. |
| `SerializeTensor(BinaryWriter,Tensor<>)` | Serializes a tensor to a binary stream. |
| `SerializeVector(BinaryWriter,Vector<>)` | Serializes a vector to a binary format. |
| `SerializeVector(Vector<>)` | Serializes a vector to a byte array. |
| `WriteValue(BinaryWriter,)` | Writes a value of type T to a binary writer. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Provides operations specific to the numeric type being used. |

