---
title: "PrincipalComponentRegressionOptions<T>"
description: "Configuration options for Principal Component Regression (PCR), which combines principal component analysis with linear regression to address multicollinearity and dimensionality issues in regression problems."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Principal Component Regression (PCR), which combines principal component analysis
with linear regression to address multicollinearity and dimensionality issues in regression problems.

## For Beginners

Principal Component Regression helps solve problems with complex, highly related data.

Imagine you're trying to predict house prices with 50 different variables:

- Many of these variables are strongly related (like number of rooms, square footage, number of bathrooms)
- Using all these related variables directly can confuse your model
- The model might become unstable or "overfit" to your training data

What Principal Component Regression does:

Step 1: Principal Component Analysis (PCA)

- It combines your original variables into new "super variables" called principal components
- Each component captures a different pattern in your data
- The first component captures the strongest pattern, the second component the next strongest, and so on
- These components are completely unrelated to each other (uncorrelated)

Step 2: Regression

- Instead of using your original 50 variables, it uses the top principal components
- This makes your model more stable and often more accurate

Think of it like cooking:

- Your original variables are like individual spices
- PCA combines these into a few special spice mixes (components)
- Your recipe now uses these few special mixes instead of dozens of individual spices
- This makes cooking (modeling) simpler and often gives better results

This class lets you configure how many components to use and how much information to retain.

## How It Works

Principal Component Regression (PCR) is a two-step technique that first uses Principal Component Analysis (PCA)
to reduce the dimensionality of the feature space, and then performs linear regression on the resulting principal
components. This approach is particularly valuable when dealing with datasets where the predictor variables are
highly correlated (multicollinearity) or when the number of predictors is large relative to the number of
observations. By transforming the original features into uncorrelated principal components, PCR mitigates issues
such as model instability and overfitting that can arise in standard regression. The reduced dimensionality
also improves computational efficiency and interpretability. PCR is widely used in fields such as spectroscopy,
chemometrics, bioinformatics, and econometrics, where high-dimensional, correlated data is common.

## Properties

| Property | Summary |
|:-----|:--------|
| `ExplainedVarianceRatio` | Gets or sets the minimum ratio of variance to be explained by the selected principal components. |
| `NumComponents` | Gets or sets the number of principal components to use in the regression model. |

