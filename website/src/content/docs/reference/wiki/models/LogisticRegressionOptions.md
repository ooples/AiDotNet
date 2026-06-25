---
title: "LogisticRegressionOptions<T>"
description: "Configuration options for Logistic Regression, a statistical method used for binary classification problems in machine learning."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Logistic Regression, a statistical method used for binary
classification problems in machine learning.

## For Beginners

Logistic Regression is one of the most fundamental classification
algorithms in machine learning, used when you want to predict categories (like "yes/no" or "spam/not spam").

Despite having "regression" in its name, it's actually used for classification problems:

- It calculates the probability that something belongs to a particular category
- If the probability is above 50%, it predicts one category; otherwise, it predicts the other

Think of it like determining whether a student will pass or fail an exam:

- You gather information about study hours, attendance, and previous grades
- Logistic Regression learns how these factors contribute to passing probability
- When a new student comes along, you can predict their outcome based on these factors

This class allows you to configure how the algorithm learns these relationships from your data.

## How It Works

Logistic Regression is a supervised learning algorithm that predicts the probability of an
observation belonging to a certain class. Despite its name, it's used for classification rather
than regression. The algorithm applies the logistic function to a linear combination of features
to transform the output to a probability value between 0 and 1. The model parameters are typically
learned through an iterative optimization process like gradient descent, which aims to maximize
the likelihood of the observed data.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate that controls the step size in each iteration of the optimization. |
| `MaxIterations` | Gets or sets the maximum number of iterations allowed for the optimization algorithm. |
| `Tolerance` | Gets or sets the convergence tolerance that determines when the optimization algorithm should stop. |

