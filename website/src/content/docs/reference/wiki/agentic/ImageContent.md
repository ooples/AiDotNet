---
title: "ImageContent"
description: "An image content part within a `ChatMessage`, supplied either as raw bytes or as a URI."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Agentic.Models`

An image content part within a `ChatMessage`, supplied either as raw bytes or as a URI.

## For Beginners

If you want the model to "look at" a picture, wrap it in one of these.
You either hand over the actual image bytes (e.g., a PNG you loaded from disk) via
`ImageMediaType)`, or point at a web address via `ImageMediaType})`. The image type is
chosen from the `ImageMediaType` enum, so you can't misspell it.

## How It Works

Multimodal models accept images alongside text. An image can be embedded directly as bytes
(which providers typically base64-encode on the wire) or referenced by a URL the provider fetches.
The format is a type-safe `ImageMediaType` rather than a raw MIME string. Use the
`ImageMediaType)` and `ImageMediaType})` factories to construct instances.

## Properties

| Property | Summary |
|:-----|:--------|
| `Data` | Gets the raw image bytes, or `null` when the image is referenced by `Uri`. |
| `HasData` | Gets a value indicating whether the image is supplied inline as bytes (rather than by URI). |
| `MediaType` | Gets the image format, or `null` when referenced by URI with an unspecified format. |
| `Uri` | Gets the image URI, or `null` when the image is supplied inline via `Data`. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromBytes(Byte[],ImageMediaType)` | Creates an image content part from raw bytes. |
| `FromUri(String,Nullable<ImageMediaType>)` | Creates an image content part that references an image by URI. |

