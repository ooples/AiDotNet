---
title: "TableStructureResult<T>"
description: "Represents the result of table structure recognition."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Document`

Represents the result of table structure recognition.

## For Beginners

Tables have structure (rows, columns, cells) and content.
This result class describes the table's layout including merged cells,
headers, and the data in each cell.

## Properties

| Property | Summary |
|:-----|:--------|
| `Cells` | Gets all cells in the table. |
| `Confidence` | Gets the overall confidence for the structure detection. |
| `HasBorders` | Gets whether the table has detected borders. |
| `HeaderRows` | Gets the header row indices (often row 0). |
| `NumColumns` | Gets the number of columns in the table. |
| `NumRows` | Gets the number of rows in the table. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ToStringGrid` | Gets the table content as a 2D list of strings. |

