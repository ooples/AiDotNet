---
title: "ColumnMetadata"
description: "Describes the metadata for a single column in a tabular dataset, including its name, data type, categories (for categorical columns), and summary statistics."
section: "API Reference"
---

`Models & Types` · `AiDotNet.NeuralNetworks.SyntheticData`

Describes the metadata for a single column in a tabular dataset, including its name,
data type, categories (for categorical columns), and summary statistics.

## For Beginners

Think of this as a "column profile" that describes everything
the generator needs to know about one column in your table:

- **Name**: A human-readable label (e.g., "Age", "Income")
- **DataType**: Whether it's continuous, discrete, or categorical
- **Categories**: For categorical columns, the list of possible values
- **Statistics**: Min, max, mean, and standard deviation (computed during fitting)

Example:

## How It Works

Column metadata is used by synthetic data generators to understand the structure of each
column and apply the appropriate preprocessing (e.g., VGM normalization for continuous,
one-hot encoding for categorical). Statistics are populated during the `Fit` step.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ColumnMetadata(String,ColumnDataType,IEnumerable<String>,Int32)` | Initializes a new instance of the `ColumnMetadata` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Categories` | Gets or sets the list of category values for categorical columns. |
| `ColumnIndex` | Gets or sets the index of this column in the original data matrix. |
| `DataType` | Gets or sets the data type of the column. |
| `IsCategorical` | Gets whether this column is categorical. |
| `IsNumerical` | Gets whether this column is numerical (continuous or discrete). |
| `Max` | Gets or sets the maximum observed value for numerical columns. |
| `Mean` | Gets or sets the mean (average) value for numerical columns. |
| `Min` | Gets or sets the minimum observed value for numerical columns. |
| `Name` | Gets or sets the name of the column. |
| `NumCategories` | Gets the number of categories for categorical columns. |
| `Std` | Gets or sets the standard deviation for numerical columns. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Clone` | Creates a deep copy of this column metadata. |
| `Deserialize(BinaryReader)` | Reads a column's metadata previously written by `BinaryWriter)`. |
| `Serialize(BinaryWriter)` | Writes this column's metadata (name, type, categories, statistics, index) to a binary stream. |

