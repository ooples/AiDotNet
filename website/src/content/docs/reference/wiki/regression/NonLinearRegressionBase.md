---
title: "NonLinearRegressionBase"
description: "Base class for non-linear regression algorithms that provides common functionality for training and prediction."
section: "Reference"
---

_Regression Models_

Base class for non-linear regression algorithms that provides common functionality for training and prediction.

## How It Works

This abstract class implements core functionality shared by different non-linear regression algorithms,
including kernel functions, regularization, and model serialization/deserialization.

Non-linear regression models can capture complex relationships in data that linear models cannot represent.
They typically use kernel functions to transform the input space into a higher-dimensional feature space
where the relationship becomes linear.

For Beginners:
Non-linear regression is used when your data doesn't follow a straight line pattern. These models can
capture curved or complex relationships between your input features and target values. Think of it like
having a flexible curve that can bend and shape itself to fit your data points, rather than just a
straight line.

