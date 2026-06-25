---
title: "KNearestNeighborsOptions"
description: "Configuration options for the K-Nearest Neighbors algorithm, which makes predictions based on the values of the K closest data points in the training set."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the K-Nearest Neighbors algorithm, which makes predictions based on the
values of the K closest data points in the training set.

## For Beginners

K-Nearest Neighbors (KNN) is one of the simplest machine learning algorithms
to understand. It works on a very intuitive principle: things that are similar tend to have similar
outcomes.

Imagine you want to predict the price of a house. With KNN, you would:

1. Find the K houses in your training data that are most similar to the house you're trying to price

(based on features like size, location, number of bedrooms, etc.)

2. Take the average price of those K houses as your prediction

The "K" in KNN is simply how many neighbors you consider when making your prediction. If K=5, you look
at the 5 most similar houses.

KNN is different from many other algorithms because it doesn't build a complex model during training.
Instead, it simply remembers all the training examples and uses them directly when making predictions.
This makes it easy to understand but can make it slower for predictions on large datasets.

This class inherits from NonLinearRegressionOptions, so all the general non-linear regression settings
are also available. The additional setting specific to KNN lets you configure how many neighbors to
consider when making predictions.

## How It Works

K-Nearest Neighbors (KNN) is a simple but powerful non-parametric algorithm that can be used for both
classification and regression. For regression, it predicts the value of a new data point by averaging
the values of its K nearest neighbors in the training data. The algorithm doesn't build an explicit
model during training - instead, it stores the training data and performs calculations at prediction time,
making it an example of "lazy learning."

## Properties

| Property | Summary |
|:-----|:--------|
| `K` | Gets or sets the number of nearest neighbors to consider when making predictions. |

