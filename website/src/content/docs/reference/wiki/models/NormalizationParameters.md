---
title: "NormalizationParameters<T>"
description: "Represents the parameters used for normalizing a single feature or target variable in a machine learning model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents the parameters used for normalizing a single feature or target variable in a machine learning model.

## For Beginners

This class stores the information needed to scale a single feature or target variable.

When normalizing data for machine learning:

- Different methods can be used (min-max scaling, z-score normalization, etc.)
- Each method requires specific parameters (like minimum/maximum values or mean/standard deviation)
- These parameters need to be saved to ensure consistent scaling

This class stores all those parameters for a single feature, including:

- Which normalization method is being used
- The specific values needed for that method (min/max, mean/stddev, etc.)

For example, if using min-max scaling to normalize house prices from $100,000-$1,500,000 to a 0-1 range,
this class would store the minimum ($100,000) and maximum ($1,500,000) values needed for that conversion.

## How It Works

This class encapsulates all the parameters needed to normalize and denormalize a single feature or target variable. 
It supports multiple normalization methods, such as min-max scaling, z-score normalization, robust scaling, and 
binning, and stores the relevant parameters for each method. These parameters are typically calculated during 
training based on the training data and are then used to normalize new data in the same way.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `NormalizationParameters(INumericOperations<>)` | Initializes a new instance of the NormalizationParameters class with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IQR` | Gets or sets the interquartile range (IQR) of the data. |
| `Max` | Gets or sets the maximum value observed in the data. |
| `MaxAbs` | Gets or sets the maximum absolute value observed in the data. |
| `Mean` | Gets or sets the mean (average) value of the data. |
| `Median` | Gets or sets the median value of the data. |
| `Method` | Gets or sets the normalization method used. |
| `Min` | Gets or sets the minimum value observed in the data. |
| `OutputDistribution` | Gets or sets the target output distribution for quantile transformation. |
| `P` | Gets or sets a power parameter for certain normalization methods. |
| `Quantiles` | Gets or sets the quantile values used for quantile transformation. |
| `Scale` | Gets or sets the scale factor for custom normalization. |
| `Shift` | Gets or sets the shift value for custom normalization. |
| `StdDev` | Gets or sets the standard deviation of the data. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | The numeric operations provider used for mathematical operations on type T. |

