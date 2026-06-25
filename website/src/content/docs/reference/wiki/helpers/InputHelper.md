---
title: "InputHelper<T, TInput>"
description: "Provides helper methods for input-related operations."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides helper methods for input-related operations.

## Methods

| Method | Summary |
|:-----|:--------|
| `CopyTensorData(Tensor<>,Tensor<>,Int32)` | Copies tensor data into a batch tensor at the specified batch index. |
| `CreateBatchFromMatrix(Matrix<>)` | Creates a batch from a single matrix, ensuring it's in the correct batch format. |
| `CreateBatchFromScalar()` | Creates a batch from a single scalar value. |
| `CreateBatchFromTensor(Tensor<>)` | Creates a batch from a single tensor, ensuring the first dimension is 1. |
| `CreateBatchFromVector(Vector<>)` | Creates a batch from a single vector. |
| `CreateSingleItemBatch()` | Creates a batch containing a single item. |
| `GetBatch(,Int32[])` | Extracts a batch of data from the input based on the specified indices. |
| `GetBatchSize()` | Gets the batch size from the input data. |
| `GetElement(,Int32,Int32)` | Gets an element at the specified position from the input data structure. |
| `GetFeatureValue(,Int32)` | Retrieves a specific feature value from an input item. |
| `GetInputSize()` | Gets the size of the input data. |
| `GetItem(,Int32)` | Retrieves a single item from a batch of input data. |
| `GetMatrixBatch(Matrix<>,Int32[])` | Extracts a batch from a matrix based on the specified row indices. |
| `GetTensorBatch(Tensor<>,Int32[])` | Extracts a batch from a tensor based on the specified indices along the first dimension. |
| `GetVectorBatch(Vector<>,Int32[])` | Extracts a batch from a vector based on the specified indices. |

