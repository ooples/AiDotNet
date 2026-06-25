---
title: "Hdf5Dataset<T>"
description: "Provides read/write access to datasets in a custom binary format for named multidimensional arrays, inspired by the HDF5 data model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Data.Formats`

Provides read/write access to datasets in a custom binary format for named multidimensional arrays,
inspired by the HDF5 data model.

## How It Works

**Important:** This is NOT a native HDF5 implementation and is NOT compatible with
HDF5 files created by h5py, HDFView, or other HDF5 tools. It is a custom binary format
designed for storing named tensor datasets within AiDotNet. For native HDF5 interop,
use the PureHDF NuGet package.

The on-disk format:

- Header: [magic: "HDF5" 4 bytes] [version: 4 bytes] [numDatasets: 4 bytes]
- Dataset table: for each dataset: [nameLen: 4][name: bytes][rank: 4][dims: 4*rank][dataOffset: 8][dataLength: 8]
- Data section: raw double-precision values for each dataset

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Hdf5Dataset(Hdf5DatasetOptions)` | Opens an HDF5 dataset file. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DatasetNames` | Gets the names of all datasets in the file. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Dispose` |  |
| `GetShape(String)` | Gets the shape of a named dataset. |
| `ReadDataset(String)` | Reads a named dataset as a Tensor. |
| `ReadSlice(String,Int32,Int32)` | Reads a slice of rows from a named dataset. |
| `WriteFile(String,IReadOnlyDictionary<String,ValueTuple<[],Int32[]>>)` | Writes multiple named datasets to an HDF5 file. |

