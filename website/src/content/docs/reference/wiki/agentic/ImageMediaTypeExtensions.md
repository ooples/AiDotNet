---
title: "ImageMediaTypeExtensions"
description: "Conversions between `ImageMediaType` and its wire-format MIME string."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Agentic.Models`

Conversions between `ImageMediaType` and its wire-format MIME string.

## For Beginners

Providers expect image types written as `image/png`, `image/jpeg`,
etc. These helpers translate between the type-safe enum your code uses and that wire text.

## Methods

| Method | Summary |
|:-----|:--------|
| `ToMimeType(ImageMediaType)` | Converts an `ImageMediaType` to its MIME string (e.g. |
| `TryParseMimeType(String,ImageMediaType)` | Parses a MIME string into an `ImageMediaType`. |

