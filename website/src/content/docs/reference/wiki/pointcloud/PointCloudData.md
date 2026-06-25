---
title: "PointCloudData<T>"
description: "Represents a point cloud data structure with coordinates and optional features."
section: "API Reference"
---

`Models & Types` · `AiDotNet.PointCloud.Data`

Represents a point cloud data structure with coordinates and optional features.

## How It Works

**For Beginners:** This class stores 3D point cloud data in an organized way.

A point cloud consists of:

- Points: XYZ coordinates representing positions in 3D space
- Features: Optional additional information per point (color, intensity, normals, etc.)
- Labels: Optional category or class information for each point

Think of it as a spreadsheet where:

- Each row is a point
- First 3 columns are X, Y, Z coordinates
- Additional columns can store colors, surface properties, etc.
- Another column can store what category each point belongs to

This structure makes it easy to:

- Load point cloud data from sensors or files
- Pass data to neural networks for processing
- Store results from classification or segmentation

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PointCloudData(Tensor<>,Vector<>)` | Initializes a new instance of the PointCloudData class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Labels` | Gets or sets optional labels for classification or segmentation tasks. |
| `NumFeatures` | Gets or sets the number of features per point (including XYZ coordinates). |
| `NumPoints` | Gets or sets the number of points in the cloud. |
| `Points` | Gets or sets the tensor containing point coordinates and features. |

## Methods

| Method | Summary |
|:-----|:--------|
| `FromCoordinates(Matrix<>,Vector<>)` | Creates a point cloud from coordinates only (no additional features). |
| `GetCoordinates` | Extracts only the XYZ coordinates from the point cloud. |
| `GetFeatures` | Extracts additional features (excluding XYZ coordinates). |

