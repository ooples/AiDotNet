---
title: "ConditionalInferenceTreeOptions"
description: "Configuration options for Conditional Inference Trees, a statistically-driven approach to decision tree learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Conditional Inference Trees, a statistically-driven approach to decision tree learning.

## For Beginners

Conditional Inference Trees are a special type of decision tree that uses 
statistics to make better decisions about how to split data. Regular decision trees sometimes favor 
certain types of data unfairly (like preferring variables with more possible values). This approach is 
like having a referee that makes sure the tree-building process is fair and statistically sound. 
The result is often a more reliable model, especially for data where some variables have many possible 
values and others have few.

## How It Works

Conditional Inference Trees use statistical tests to select variables and determine split points,
which helps reduce selection bias toward variables with many possible split points.
This approach often produces more reliable and statistically sound trees compared to traditional methods.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDegreeOfParallelism` | Gets or sets the maximum number of parallel operations when building the tree. |
| `SignificanceLevel` | Gets or sets the statistical significance level used for hypothesis testing when selecting split variables. |
| `StatisticalTest` | Gets or sets the type of statistical test used to evaluate potential splits. |

