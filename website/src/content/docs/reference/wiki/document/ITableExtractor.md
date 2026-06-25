---
title: "ITableExtractor<T>"
description: "Interface for table detection and structure recognition models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Document.Interfaces`

Interface for table detection and structure recognition models.

## For Beginners

Documents often contain tables with important data. Table extraction
helps computers understand where tables are, how they're structured, and what data they contain.
This is useful for extracting financial data, product catalogs, or any tabular information.

Example usage:

## How It Works

Table extraction models detect tables in documents and extract their structure
(rows, columns, cells) along with the content in each cell.

## Properties

| Property | Summary |
|:-----|:--------|
| `SupportsBorderedTables` | Gets whether this model supports bordered tables. |
| `SupportsBorderlessTables` | Gets whether this model supports borderless tables. |
| `SupportsMergedCells` | Gets whether this model can detect merged cells (row/column spans). |

## Methods

| Method | Summary |
|:-----|:--------|
| `DetectTables(Tensor<>)` | Detects tables in a document image. |
| `ExportTables(Tensor<>,TableExportFormat)` | Exports detected tables to a specific format. |
| `ExtractTableContent(Tensor<>)` | Extracts table content as structured data. |
| `RecognizeStructure(Tensor<>)` | Recognizes the structure of a table (rows, columns, cells). |

