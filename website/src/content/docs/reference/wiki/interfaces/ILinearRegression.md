---
title: "ILinearRegression<T>"
description: "Defines an interface for linear regression in machine learning, which predict outputs as a weighted sum of inputs plus an optional constant."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for linear regression in machine learning, which predict outputs as a weighted sum of inputs plus an optional constant.

## How It Works

**For Beginners:** This interface represents the simplest and most fundamental type of machine learning model.

Imagine you're trying to predict house prices based on features like:

- Square footage
- Number of bedrooms
- Age of the house

A linear model works like this:

- Each feature gets assigned a "weight" (coefficient) that represents its importance
- For example: Each square foot might add $100 to the price
- Each bedroom might add $15,000 to the price
- Each year of age might subtract $500 from the price
- There might also be a "starting price" (intercept) of $50,000

To make a prediction, the model:

1. Multiplies each feature by its weight
2. Adds all these values together
3. Adds the intercept (if there is one)

So for a 2,000 sq ft, 3-bedroom, 10-year-old house:
Price = $50,000 + (2,000 × $100) + (3 × $15,000) + (10 × -$500)
Price = $50,000 + $200,000 + $45,000 - $5,000
Price = $290,000

Linear models are popular because they're:

- Simple to understand
- Fast to train
- Easy to interpret (you can see exactly how each feature affects the prediction)
- Often surprisingly effective despite their simplicity

## Properties

| Property | Summary |
|:-----|:--------|
| `Coefficients` | Gets the weights (coefficients) assigned to each input feature in the linear model. |
| `HasIntercept` | Gets a value indicating whether this linear model includes an intercept term. |
| `Intercept` | Gets the constant term (bias) added to the weighted sum of features in the linear model. |

