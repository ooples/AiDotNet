---
title: "MultinomialLogisticRegressionOptions<T>"
description: "Configuration options for Multinomial Logistic Regression, a classification method that generalizes logistic regression to multiclass problems with more than two possible discrete outcomes."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Multinomial Logistic Regression, a classification method that generalizes
logistic regression to multiclass problems with more than two possible discrete outcomes.

## For Beginners

Multinomial Logistic Regression is a technique for classifying data into
multiple categories.

While regular Logistic Regression can only decide between two options (like "yes" or "no"),
Multinomial Logistic Regression can decide between many options - for example:

- Classifying emails as "work," "personal," or "spam"
- Identifying handwritten digits (0-9)
- Categorizing products into different types

Think of it like a voting system:

- Each feature in your data "votes" for different categories
- The model learns how much weight to give each feature's vote
- When making a prediction, it collects all these weighted votes 
- It converts these votes into probabilities for each category using a "softmax" function
- The category with the highest probability is the prediction

This class lets you configure how the model learns these voting weights from your training data.

## How It Works

Multinomial Logistic Regression extends binary logistic regression to handle multiple classes by using
the softmax function rather than the sigmoid function. It models the probabilities of each class directly,
learning a set of coefficients for each class (except one reference class). The model is trained using
maximum likelihood estimation, typically optimized through iterative methods like gradient descent or
Newton's method. This approach is also known as Softmax Regression or Maximum Entropy Classification.

## Properties

| Property | Summary |
|:-----|:--------|
| `DecompositionType` | Gets or sets the matrix decomposition type to use when solving the linear system. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed for the optimization algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance that determines when the optimization algorithm should stop. |

