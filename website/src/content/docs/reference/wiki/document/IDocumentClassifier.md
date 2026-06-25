---
title: "IDocumentClassifier<T>"
description: "Interface for document classification models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for document classification models.

## For Beginners

Document classification is like sorting mail into different piles.
The model looks at a document and decides what type it is. This is useful for:

- Organizing scanned documents
- Routing documents to appropriate processing pipelines
- Quality control in document processing

Example usage:

## How It Works

Document classification models categorize documents into predefined classes
such as invoices, forms, letters, scientific papers, etc.

## Properties

| Property | Summary |
|:-----|:--------|
| `AvailableCategories` | Gets the available classification categories for this model. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ClassifyDocument(Tensor<>)` | Classifies a document image into predefined categories. |
| `ClassifyDocument(Tensor<>,Int32)` | Classifies a document and returns top-K predictions. |

