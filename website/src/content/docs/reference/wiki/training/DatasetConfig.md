---
title: "DatasetConfig"
description: "Configuration for the dataset section of a training recipe."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Training.Configuration`

Configuration for the dataset section of a training recipe.

## For Beginners

This defines where your data comes from and how it should be loaded.
You can specify a CSV file path, whether it has headers, the batch size for training,
and which column contains the labels (the values you want to predict).

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchSize` | Gets or sets the number of samples per training batch. |
| `HasHeader` | Gets or sets whether the CSV file has a header row. |
| `LabelColumn` | Gets or sets the zero-based index of the label column. |
| `Name` | Gets or sets an optional descriptive name for the dataset. |
| `Path` | Gets or sets the file path to the dataset (currently supports CSV files). |

