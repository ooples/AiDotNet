---
title: "DistributionFitResult<T>"
description: "Represents the result of fitting a statistical distribution to a dataset, including the distribution type, goodness of fit measure, and estimated parameters."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Represents the result of fitting a statistical distribution to a dataset, including the distribution type,
goodness of fit measure, and estimated parameters.

## For Beginners

This class stores information about how well a statistical distribution matches your data.

When analyzing data, you often want to know:

- Which standard statistical distribution (like Normal, Exponential, etc.) best describes your data
- How well that distribution fits your data
- What the specific parameters of that distribution are

This class stores all that information after a distribution fitting process, making it easy to:

- Identify the best distribution for your data
- Compare how well different distributions fit
- Access the parameters needed to use the distribution for further analysis

For example, if your data follows a normal distribution, this class would tell you that,
provide a measure of how well it fits, and give you the mean and standard deviation parameters.

## How It Works

When analyzing data, it's often useful to determine which statistical distribution best describes the data. 
This class stores the results of such distribution fitting, including the type of distribution that best fits 
the data, a measure of how well the distribution fits (goodness of fit), and the estimated parameters of the 
distribution. The goodness of fit measure allows for comparing different distribution types to determine which 
one provides the best representation of the data. The class uses generic type parameter T to support different 
numeric types for the statistical values, such as float, double, or decimal.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DistributionFitResult(INumericOperations<>)` | Initializes a new instance of the DistributionFitResult class with default values. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistributionType` | Gets or sets the type of distribution that best fits the data. |
| `GoodnessOfFit` | Gets or sets the goodness of fit measure. |
| `Parameters` | Gets or sets the parameters of the fitted distribution. |

