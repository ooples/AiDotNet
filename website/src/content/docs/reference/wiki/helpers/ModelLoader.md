---
title: "ModelLoader"
description: "Provides static methods for loading self-describing AIMF model files with automatic type detection."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides static methods for loading self-describing AIMF model files with automatic type detection.

## How It Works

**For Beginners:** When you save a model using the new AIMF envelope format, the file contains
metadata about what type of model it is. ModelLoader reads this metadata and automatically creates
the correct model instance, so you don't need to know the exact model type in advance.

Example usage:

All model files must use the AIMF envelope format. Files saved with SaveModel()
automatically include the AIMF header.

## Methods

| Method | Summary |
|:-----|:--------|
| `Inspect(String)` | Reads only the header of an AIMF model file without loading the full model. |
| `IsSelfDescribing(String)` | Checks whether a file is a self-describing AIMF model file by reading only the first 4 bytes. |
| `Load(String,String,Byte[])` | Loads a self-describing AIMF model file, automatically detecting and instantiating the correct model type. |
| `LoadFromBytes(Byte[],String,Byte[])` | Loads a self-describing AIMF model from a byte array. |
| `SaveEncrypted(IModelSerializer,String,String,Int32[],Int32[],SerializationFormat,DynamicShapeInfo,Byte[])` | Saves a model to an encrypted AIMF file that requires a license key to load. |
| `ValidateShapes(IModelShape,ModelFileInfo)` | Validates that the deserialized model's shapes match those stored in the header. |

