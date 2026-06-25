---
title: "OrthogonalRegressionOptions<T>"
description: "Configuration options for Orthogonal Regression (also known as Total Least Squares), which minimizes  the perpendicular distances from data points to the fitted model, accounting for errors in both  dependent and independent variables."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Orthogonal Regression (also known as Total Least Squares), which minimizes 
the perpendicular distances from data points to the fitted model, accounting for errors in both 
dependent and independent variables.

## For Beginners

Orthogonal Regression is a special type of regression that treats all variables fairly when finding patterns.

In standard regression:

- We assume that only the y-variable (what we're predicting) contains errors
- We minimize the vertical distances from points to the line

Imagine measuring the heights and weights of people:

- Standard regression assumes heights are measured perfectly, only weights have errors
- Orthogonal regression recognizes that both height AND weight measurements have errors

This matters because:

- When both variables have measurement errors, standard regression can give biased results
- Orthogonal regression fits a line that's "fair" to both variables
- The line minimizes the perpendicular distance from points to the line, not just vertical distance

This technique is especially useful in scientific applications where:

- All measurements come from instruments with known error rates
- We're looking for true physical relationships rather than just predictions
- The variables play symmetrical roles rather than strictly "input" and "output"

This class lets you configure how the orthogonal regression algorithm works, controlling
its precision, computational limits, and data preprocessing.

## How It Works

Orthogonal Regression differs from standard regression techniques by considering measurement errors 
in both the predictor (independent) and response (dependent) variables. While ordinary least squares 
regression minimizes vertical distances from points to the regression line, orthogonal regression 
minimizes perpendicular distances, making it more appropriate when both variables contain measurement 
error or uncertainty. This approach is particularly valuable in fields like physics, chemistry, and 
engineering where measurement instruments may introduce errors in all variables. The algorithm typically 
employs singular value decomposition or iterative methods to find the optimal solution.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type to use when solving the linear system. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed for the optimization algorithm. |
| `ScaleVariables` | Gets or sets whether to standardize variables before fitting the model. |
| `Tolerance` | Gets or sets the convergence tolerance that determines when the iterative optimization algorithm should stop. |

