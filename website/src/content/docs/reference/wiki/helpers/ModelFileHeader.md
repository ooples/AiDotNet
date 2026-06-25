---
title: "ModelFileHeader"
description: "Provides methods for reading and writing the AIMF (AI Model File) binary envelope header."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides methods for reading and writing the AIMF (AI Model File) binary envelope header.

## How It Works

**For Beginners:** When you save a machine learning model, this helper wraps the model data
with a small header that describes what type of model it is, its input/output shapes, and
what format the data is in. This allows tools to identify and load models automatically
without needing to know the model type in advance.

The envelope format (v1) is:

The existing Serialize() output is stored unchanged as the payload (or encrypted).
The header is always plaintext, allowing Inspect() to read metadata without a key.

## Methods

| Method | Summary |
|:-----|:--------|
| `ExtractPayload(Byte[],ModelFileInfo)` | Extracts the model payload from data that has an AIMF envelope header. |
| `HasHeader(Byte[])` | Checks whether the given data starts with the AIMF magic bytes. |
| `HasHeader(String)` | Checks whether a file starts with the AIMF magic bytes by reading only the first 4 bytes. |
| `ReadHeader(Byte[])` | Reads the AIMF envelope header from a byte array. |
| `WrapWithHeader(Byte[],IModelSerializer,Int32[],Int32[],SerializationFormat,DynamicShapeInfo)` | Wraps serialized model data with an AIMF envelope header. |
| `WrapWithHeaderEncrypted(Byte[],IModelSerializer,Int32[],Int32[],SerializationFormat,Byte[],Byte[],Byte[],DynamicShapeInfo)` | Wraps serialized model data with an AIMF envelope header that includes encryption metadata. |
| `WrapWithHeaderEncrypted(Byte[],IModelSerializer,Int32[],Int32[],SerializationFormat,Byte[],Byte[],Byte[],PayloadEncryptionScheme,DynamicShapeInfo)` | Wraps an encrypted payload with an AIMF header using a specified encryption scheme. |

## Fields

| Field | Summary |
|:-----|:--------|
| `AimfMagic` | Magic bytes identifying an AIMF envelope: 0x41 0x49 0x4D 0x46 = "AIMF". |
| `CurrentEnvelopeVersion` | Current envelope version. |

