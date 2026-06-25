---
title: "ModelFileInfo"
description: "Contains parsed metadata from an AIMF (AI Model File) envelope header."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Helpers`

Contains parsed metadata from an AIMF (AI Model File) envelope header.

## How It Works

**For Beginners:** When a model file is saved with the AIMF envelope, the header contains
metadata that describes the model without needing to load it fully. This record holds
all the information extracted from that header.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelFileInfo(Int32,SerializationFormat,String,String,Int32[],Int32[],Int64,Int32,DynamicShapeInfo,PayloadEncryptionScheme,Byte[],Byte[],Byte[])` | Creates a new ModelFileInfo with the specified header values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AssemblyQualifiedName` | Gets the assembly-qualified type name for fallback resolution. |
| `DynamicShapeInfo` | Gets the dynamic shape information describing which dimensions are variable. |
| `EncryptionScheme` | Gets the encryption scheme applied to the payload. |
| `EnvelopeVersion` | Gets the envelope format version. |
| `Format` | Gets the serialization format of the model payload. |
| `HeaderLength` | Gets the byte offset where the payload starts in the original data. |
| `InputShape` | Gets the input shape of the model, or empty array if not available. |
| `IsEncrypted` | Gets whether the payload is encrypted and requires a license key to load. |
| `Nonce` | Gets the AES-GCM nonce (IV) when the payload is encrypted, or null if unencrypted. |
| `OutputShape` | Gets the output shape of the model, or empty array if not available. |
| `PayloadLength` | Gets the length of the model payload in bytes. |
| `Salt` | Gets the PBKDF2 salt used for key derivation when the payload is encrypted, or null if unencrypted. |
| `Tag` | Gets the AES-GCM authentication tag when the payload is encrypted, or null if unencrypted. |
| `TypeName` | Gets the short type name of the model (e.g., "ConvolutionalNeuralNetwork`1"). |

