---
title: "RegressionOptions<T>"
description: "Configuration options for regression models, which are statistical methods used to estimate  relationships between variables and make predictions."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for regression models, which are statistical methods used to estimate 
relationships between variables and make predictions.

## For Beginners

Regression is a way to find patterns in your data that help predict numbers.

Think about predicting house prices:

- You have information like square footage, number of bedrooms, and neighborhood
- Regression helps you create a formula that uses these features to predict the price
- The formula might look like: Price = (Square Footage × Factor1) + (Bedrooms × Factor2) + BaseValue

What regression does:

- It analyzes your existing data (houses with known prices)
- It finds the best values for each factor in the formula
- It creates a model that can predict prices for new houses

This is useful when:

- You need to predict a numerical value (like price, temperature, or sales)
- You have data with features that might influence that value
- You want to understand how much each feature contributes to the prediction

For example, a weather app might use regression to predict tomorrow's temperature based on 
today's readings, humidity levels, and seasonal patterns.

This class lets you configure how the regression model is constructed and calculated.

## How It Works

Regression analysis is a fundamental statistical technique used to model the relationship between a 
dependent variable and one or more independent variables. This class provides configuration options 
for regression models, allowing customization of how the regression algorithm operates. Regression 
models are widely used in machine learning and statistics for prediction, forecasting, and understanding 
variable relationships. Common regression types include linear regression, polynomial regression, 
and regularized regression methods like ridge and lasso regression.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionMethod` | Gets or sets the matrix decomposition method used in the regression calculations. |
| `UseIntercept` | Gets or sets whether the regression model should include an intercept term (also known as bias term). |

