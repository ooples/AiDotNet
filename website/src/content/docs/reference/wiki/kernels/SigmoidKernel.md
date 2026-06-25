---
title: "SigmoidKernel<T>"
description: "Implements the Sigmoid kernel for measuring similarity between data points."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Sigmoid kernel for measuring similarity between data points.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Sigmoid kernel is special because it can capture certain types of relationships that other
kernels can't. It's inspired by how neurons in artificial neural networks work.

## How It Works

The Sigmoid kernel is derived from the neural network field and is related to the activation
function used in artificial neural networks. It can be used to model certain types of non-linear
relationships in data.

Think of it like this: The Sigmoid kernel looks at how two data points interact with each other
(their dot product) and then transforms this interaction using a special S-shaped curve (the hyperbolic
tangent function). This helps it detect complex patterns in your data.

The formula for the Sigmoid kernel is:
k(x, y) = tanh(a(x·y) + c)
where:

- x and y are the two data points being compared
- x·y is the dot product between them
- a (alpha) controls the steepness of the S-curve
- c is a parameter that shifts the curve horizontally
- tanh is the hyperbolic tangent function (an S-shaped curve)

Common uses include:

- Text classification problems
- Some types of image recognition tasks
- Problems where the data might have complex, non-linear relationships

Note: Unlike many other kernels, the Sigmoid kernel is not always positive definite, which means
it might not work well with all machine learning algorithms. It's most commonly used with
Support Vector Machines.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SigmoidKernel(,)` | Initializes a new instance of the Sigmoid kernel with optional scaling and shifting parameters. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Sigmoid kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_alpha` | The scaling parameter that controls the steepness of the sigmoid curve. |
| `_c` | The shifting parameter that moves the sigmoid curve horizontally. |
| `_numOps` | Operations for performing numeric calculations with type T. |

