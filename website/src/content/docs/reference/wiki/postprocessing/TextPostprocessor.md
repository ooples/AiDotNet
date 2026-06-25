---
title: "TextPostprocessor<T>"
description: "TextPostprocessor - OCR text postprocessing utilities."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Postprocessing.Document`

TextPostprocessor - OCR text postprocessing utilities.

## For Beginners

OCR output often contains errors and formatting issues.
This tool cleans up the text:

- Remove unwanted characters
- Fix common OCR errors
- Normalize whitespace
- Correct formatting

Key features:

- Character normalization
- Whitespace handling
- Common OCR error correction
- Language-aware processing

Example usage:

## How It Works

TextPostprocessor provides a comprehensive pipeline for cleaning and correcting
text output from OCR systems, improving readability and accuracy.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TextPostprocessor` | Creates a new TextPostprocessor with default options. |
| `TextPostprocessor(TextPostprocessorOptions)` | Creates a new TextPostprocessor with specified options. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverse` | Text postprocessor supports inverse transformation (returns original). |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the text postprocessor. |
| `ExtractParagraphs(String)` | Extracts paragraphs from processed text. |
| `ExtractSentences(String)` | Extracts sentences from processed text. |
| `FixCommonOcrErrors(String)` | Fixes common OCR recognition errors. |
| `MergeBrokenLines(String)` | Merges lines that were incorrectly broken. |
| `NormalizeCharacters(String)` | Normalizes special characters to ASCII equivalents. |
| `NormalizeWhitespace(String)` | Normalizes whitespace in the text. |
| `ProcessCore(String)` | Processes OCR text through the full postprocessing pipeline. |
| `RemoveControlCharacters(String)` | Removes control characters from text. |
| `RemoveDuplicateSpaces(String)` | Removes duplicate consecutive spaces. |
| `RemoveHeadersFooters(String,Int32,Int32)` | Removes headers and footers from document text. |
| `RemovePageNumbers(String)` | Removes page numbers from text. |
| `ValidateInput(String)` | Validates the input text. |

