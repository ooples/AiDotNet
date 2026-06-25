---
title: "IAsyncTreeBasedModel<T>"
description: "Defines an interface for asynchronous tree-based machine learning models."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines an interface for asynchronous tree-based machine learning models.

## How It Works

**For Beginners:** This interface extends the regular tree-based model interface to add
asynchronous (async) capabilities.

Tree-based models are machine learning algorithms that make decisions using a tree-like
structure of questions - similar to a flowchart. Popular examples include Decision Trees,
Random Forests, and Gradient Boosting Trees.

"Asynchronous" (or "async") means the model can run in the background without blocking
other operations. This is especially useful for:

- Training large models that take a long time
- Working with web applications where you don't want to freeze the user interface
- Processing large datasets efficiently

Think of it like ordering food at a restaurant - instead of standing at the counter
waiting for your order (synchronous), you get a buzzer and can do other things until
your food is ready (asynchronous).

This interface inherits all the regular methods from ITreeBasedModel but adds async
versions of the training and prediction methods.

## Methods

| Method | Summary |
|:-----|:--------|
| `PredictAsync(Matrix<>)` | Makes predictions asynchronously using the trained model for the given input data. |
| `TrainAsync(Matrix<>,Vector<>)` | Trains the tree-based model asynchronously using the provided input features and target values. |

