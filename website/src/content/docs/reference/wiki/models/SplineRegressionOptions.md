---
title: "SplineRegressionOptions"
description: "Configuration options for spline regression models, which fit piecewise polynomial functions to data for flexible nonlinear modeling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for spline regression models, which fit piecewise polynomial functions
to data for flexible nonlinear modeling.

## For Beginners

Spline regression helps model complex relationships in your data using connected smooth curves.

When simple linear regression isn't flexible enough:

- Linear regression forces a straight line through your data
- Polynomial regression uses a single curved line, but can behave poorly
- Spline regression uses multiple connected curves for more flexibility

Think of spline regression like drawing with multiple connected curve segments:

- Each segment is a polynomial curve (like a parabola)
- The segments connect smoothly at points called "knots"
- This creates a flexible curve that can adapt to different patterns in different regions

Benefits of spline regression:

- More flexible than simple lines or polynomials
- Avoids the wild oscillations that can happen with high-degree polynomials
- Can capture complex relationships while still being relatively simple to interpret

This class lets you configure exactly how the spline regression model will be constructed.

## How It Works

Spline regression is a flexible nonlinear regression technique that fits piecewise polynomial functions 
(splines) to data. Unlike simple polynomial regression, which uses a single polynomial for the entire 
data range, spline regression divides the data into segments and fits separate polynomials to each segment, 
with constraints ensuring smoothness at the connection points (knots). This approach provides greater 
flexibility in modeling complex nonlinear relationships while avoiding the oscillatory behavior often 
seen with high-degree polynomials. This class provides configuration options for spline regression, 
including the number of knots (which determines the number of segments), the degree of the polynomial 
functions, and the matrix decomposition method used for solving the regression equations.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition method used to solve the regression equations. |
| `Degree` | Gets or sets the degree of the polynomial functions used in each segment of the spline. |
| `NumberOfKnots` | Gets or sets the number of knots (internal breakpoints) in the spline function. |

