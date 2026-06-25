---
title: "DataPreparationPipeline<T>"
description: "Chains multiple row operations into a sequential data preparation pipeline."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation`

Chains multiple row operations into a sequential data preparation pipeline.

## For Beginners

Before training a model, you might want to:

- Remove outliers (unusual data points that could confuse the model)
- Add synthetic samples (SMOTE) to balance classes
- Split data into train/validation/test sets

This pipeline handles all these operations in sequence, making sure your features
and labels stay properly aligned.

## How It Works

DataPreparationPipeline handles operations that change the number of rows in a dataset,
such as outlier removal, data augmentation (SMOTE), and data splitting. These operations
must process both features (X) and labels (y) together to maintain alignment.

**Data Preparation vs Data Preprocessing:**

- **Data Preparation (this pipeline):** Changes the NUMBER of rows - outlier removal,

augmentation, splitting. Only happens during training.

- **Data Preprocessing:** Transforms values WITHOUT changing row count - scaling,

encoding. Happens during both training and prediction.

**Pipeline Flow:**

**Usage:** Users interact with this through AiModelBuilder.ConfigureDataPreparation().

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DataPreparationPipeline` | Creates a new empty data preparation pipeline. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Count` | Gets the number of operations in the pipeline. |
| `HasSplitter` | Gets whether this pipeline has a splitter configured. |
| `IsFitted` | Gets whether this pipeline has been fitted to data. |
| `Operations` | Gets the named operations in the pipeline. |
| `Splitter` | Gets the configured data splitter, or null if none is set. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Add(IRowOperation<>)` | Adds a row operation to the pipeline. |
| `Add(String,IRowOperation<>)` | Adds a named row operation to the pipeline. |
| `Clear` | Clears all operations and splitter from the pipeline. |
| `FitResample(Matrix<>,Vector<>)` | Fits all operations and applies row modifications to the data. |
| `FitResampleAndGetSplits(Matrix<>,Vector<>)` | Applies row operations and returns multiple splits (for cross-validation). |
| `FitResampleAndGetTensorSplits(Tensor<>,Tensor<>)` | Applies row operations and returns multiple tensor splits (for cross-validation). |
| `FitResampleAndSplit(Matrix<>,Vector<>)` | Applies row operations and then splits the data. |
| `FitResampleAndSplitTensor(Tensor<>,Tensor<>)` | Applies row operations and then splits the tensor data. |
| `FitResampleTensor(Tensor<>,Tensor<>)` | Fits all operations and applies row modifications to tensor data. |
| `GetOperation(String)` | Gets an operation by name. |
| `GetSummary` | Gets a summary of the pipeline operations and splitter. |
| `WithKFold(Int32,Boolean,Int32)` | Configures K-Fold cross-validation. |
| `WithSplitter(IDataSplitter<>)` | Configures a custom data splitter for this pipeline. |
| `WithStratifiedKFold(Int32,Boolean,Int32)` | Configures Stratified K-Fold cross-validation that preserves class distribution. |
| `WithStratifiedSplit(Double,Boolean,Int32)` | Configures a stratified train/test split for classification. |
| `WithTimeSeriesSplit(Int32,Int32)` | Configures a time series split with expanding window (no shuffling). |
| `WithTrainTestSplit(Double,Boolean,Int32)` | Configures a simple train/test split. |
| `WithTrainValTestSplit(Double,Double,Boolean,Int32)` | Configures a three-way train/validation/test split. |

