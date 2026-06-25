---
title: "StructuredOutputTemplate"
description: "Template that guides models to produce structured output in specific formats."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PromptEngineering.Templates`

Template that guides models to produce structured output in specific formats.

## For Beginners

Gets AI to output data in a specific format.

Example:

Supported formats:

- JSON for APIs and data processing
- XML for configuration and interchange
- CSV for tabular data
- Markdown for documentation
- Custom formats with user-defined schemas

## How It Works

This template helps ensure consistent, parseable output by providing
format specifications, schemas, and examples of the expected structure.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `StructuredOutputTemplate(OutputFormat,String,String)` | Initializes a new instance of the StructuredOutputTemplate class. |
| `StructuredOutputTemplate(String)` | Initializes a new instance with a custom template string. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Builder` | Creates a builder for constructing structured output templates. |
| `Csv(String[])` | Creates a CSV output template with column headers. |
| `FormatCore(Dictionary<String,String>)` | Formats the structured output template. |
| `Json(String,String)` | Creates a JSON output template with a schema. |
| `Markdown(String)` | Creates a Markdown output template. |
| `WithTask(String)` | Sets the task to perform. |
| `Xml(String,String)` | Creates an XML output template with a schema. |
| `Yaml(String)` | Creates a YAML output template with a schema. |

