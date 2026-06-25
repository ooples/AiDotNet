---
title: "DatasetCharacteristics"
description: "Represents metadata about a dataset for augmentation recommendations."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Augmentation`

Represents metadata about a dataset for augmentation recommendations.

## Properties

| Property | Summary |
|:-----|:--------|
| `CustomProperties` | Gets or sets custom characteristics as key-value pairs. |
| `HasBoundingBoxes` | Gets or sets whether the data contains bounding boxes. |
| `HasKeypoints` | Gets or sets whether the data contains keypoints. |
| `HasMasks` | Gets or sets whether the data contains segmentation masks. |
| `HasSpatialTargets` | Gets or sets whether the data contains spatial targets. |
| `HasVariableSizes` | Gets or sets whether images have varying sizes. |
| `ImageDimensions` | Gets or sets the average image dimensions (for image data). |
| `ImbalanceRatio` | Gets or sets the class imbalance ratio (max/min class count). |
| `IsImbalanced` | Gets or sets whether the dataset is imbalanced. |
| `MissingValuePercentage` | Gets or sets the percentage of missing values. |
| `NumClasses` | Gets or sets the number of classes (for classification). |
| `NumFeatures` | Gets or sets the number of features (for tabular data). |
| `SampleCount` | Gets or sets the number of samples in the dataset. |

