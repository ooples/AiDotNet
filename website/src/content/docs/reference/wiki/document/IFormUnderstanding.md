---
title: "IFormUnderstanding<T>"
description: "Interface for form understanding models that extract structured fields from documents."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for form understanding models that extract structured fields from documents.

## For Beginners

Form understanding is like having someone read a form and
fill out a digital version. The model finds:

- Field labels and their values (e.g., "Name: John Smith")
- Checkboxes and whether they're checked
- Signatures and their locations

Example usage:

## How It Works

Form understanding models extract key-value pairs, checkboxes, signatures, and
other structured information from forms, invoices, receipts, and similar documents.

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectCheckboxes(Tensor<>)` | Detects checkboxes and their states in a document. |
| `DetectSignatures(Tensor<>)` | Detects signatures in a document. |
| `ExtractFormFields(Tensor<>)` | Extracts form fields from a document image. |
| `ExtractFormFields(Tensor<>,Double)` | Extracts form fields with a custom confidence threshold. |
| `ExtractKeyValuePairs(Tensor<>)` | Extracts key-value pairs from a document. |

