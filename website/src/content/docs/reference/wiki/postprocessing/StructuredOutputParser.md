---
title: "StructuredOutputParser<T>"
description: "StructuredOutputParser - Parses document AI outputs into structured data formats."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Postprocessing.Document`

StructuredOutputParser - Parses document AI outputs into structured data formats.

## For Beginners

AI models output text, but applications need structured data.
This tool bridges that gap:

- Convert OCR text to JSON
- Extract key-value pairs
- Build tabular data
- Validate against schemas

Key features:

- Multiple output formats
- Schema validation
- Custom parsing rules
- Error handling

Example usage:

## How It Works

StructuredOutputParser converts raw model outputs and OCR results into
structured formats like JSON, tables, key-value pairs, and custom schemas.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StructuredOutputParser` | Creates a new StructuredOutputParser with default settings. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsInverse` | Structured output parser does not support inverse transformation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `Dispose(Boolean)` | Releases resources used by the parser. |
| `ParseForm(String,IEnumerable<String>)` | Parses a form into a structured object. |
| `ParseInvoice(String)` | Parses an invoice into a structured format. |
| `ParseKeyValuePairs(String)` | Parses text into key-value pairs. |
| `ParseReceipt(String)` | Parses a receipt into a structured format. |
| `ParseTable(String,String)` | Parses tabular text into a list of rows. |
| `ParseTableWithHeaders(String,String)` | Parses tabular text into a list of dictionaries using the first row as headers. |
| `ParseToJson(String,IEnumerable<FieldExtractionRule>)` | Parses text into a JSON object based on extraction rules. |
| `ProcessCore(String)` | Parses text into key-value pairs. |
| `ValidateAgainstSchema(Dictionary<String,Object>,DocumentSchema)` | Validates parsed data against a schema. |
| `ValidateInput(String)` | Validates the input text. |

