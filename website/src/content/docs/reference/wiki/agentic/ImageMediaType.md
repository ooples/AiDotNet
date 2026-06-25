---
title: "ImageMediaType"
description: "The image formats accepted by multimodal chat models."
section: "API Reference"
---

`Enums` · `AiDotNet.Agentic.Models`

The image formats accepted by multimodal chat models.

## For Beginners

Instead of passing a free-text string for the image type (easy to misspell),
you pick from a known list: PNG, JPEG, GIF, or WebP. The library turns your choice into the exact
text the provider expects.

## How It Works

Vision-capable providers accept a small, fixed set of image formats. Modeling them as an enum (rather
than a raw MIME string like `"image/png"`) prevents typos such as `"image/pngg"` from
compiling and failing at request time, and gives callers autocomplete for the valid choices. The
wire-format MIME string is derived from the enum via `ImageMediaType)`.

## Fields

| Field | Summary |
|:-----|:--------|
| `Gif` | GIF image (`image/gif`). |
| `Jpeg` | JPEG image (`image/jpeg`). |
| `Png` | PNG image (`image/png`). |
| `Webp` | WebP image (`image/webp`). |

