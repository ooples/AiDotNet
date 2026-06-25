---
title: "SupportVectorRegressionOptions"
description: "Configuration options for Support Vector Regression (SVR), a powerful regression technique that uses support vector machines to predict continuous values."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Support Vector Regression (SVR), a powerful regression technique
that uses support vector machines to predict continuous values.

## For Beginners

Support Vector Regression is a technique for predicting continuous values that works well with complex data.

When performing regression (predicting continuous values):

- Traditional methods like linear regression work well for simple relationships
- But real-world data often contains complex, non-linear patterns

Support Vector Regression solves this by:

- Creating a "tube" around the regression line/curve
- Ignoring small errors that fall within this tube
- Focusing on preventing large errors outside the tube
- Using "kernel functions" to handle non-linear relationships

This approach offers several benefits:

- Handles non-linear relationships well
- Less sensitive to outliers than many other methods
- Often generalizes well to new data
- Works effectively even with limited training data

This class lets you configure the SVR algorithm's behavior.

## How It Works

Support Vector Regression (SVR) extends the principles of Support Vector Machines (SVM) to regression 
problems. SVR works by finding a function that deviates from the observed target values by a value no 
greater than a specified margin (epsilon) for each training point, while also remaining as flat as 
possible. This approach creates an epsilon-insensitive tube around the function, ignoring errors within 
this tube. Points outside the tube become support vectors that determine the function. SVR is particularly 
effective for non-linear regression problems when combined with kernel functions, handling complex 
relationships in the data while maintaining good generalization properties. This class inherits from 
NonLinearRegressionOptions and adds parameters specific to SVR, such as the margin width (epsilon) and 
the regularization parameter (C).

## Properties

| Property | Summary |
|:-----|:--------|
| `C` | Gets or sets the regularization parameter. |
| `Epsilon` | Gets or sets the width of the epsilon-insensitive tube. |

