---
title: "ILayoutDetector<T>"
description: "Interface for document layout detection models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for document layout detection models.

## For Beginners

Think of layout detection as drawing boxes around different parts
of a document and labeling what each part is (title, paragraph, table, etc.).
This helps computers understand the structure of a document just like humans do.

Example usage:

## How It Works

Layout detection identifies and localizes different structural elements in a document,
such as text blocks, tables, figures, headers, and footers.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportedElementTypes` | Gets the layout element types this detector can identify. |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectLayout(Tensor<>)` | Detects layout regions in a document image. |
| `DetectLayout(Tensor<>,Double)` | Detects layout regions with a specified confidence threshold. |

