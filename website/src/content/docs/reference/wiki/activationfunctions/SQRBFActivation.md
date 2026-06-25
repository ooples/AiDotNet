---
title: "SQRBFActivation<T>"
description: "Implements the Squared Radial Basis Function (SQRBF) activation function."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ActivationFunctions`

Implements the Squared Radial Basis Function (SQRBF) activation function.

## For Beginners

The Squared Radial Basis Function (SQRBF) is an activation function that produces a bell-shaped curve.
Unlike functions like ReLU or Sigmoid that are used in standard neural networks, SQRBF is commonly used in 
Radial Basis Function Networks (RBFNs).

Think of SQRBF like a "proximity detector" - it gives its highest output (1.0) when the input is exactly 0,
and progressively smaller outputs as the input moves away from 0 in either direction (positive or negative).
The ß parameter controls how quickly the output drops off as you move away from 0:

- A larger ß makes the bell curve narrower (drops off quickly)
- A smaller ß makes the bell curve wider (drops off slowly)

This is useful in machine learning when you want to measure how close an input is to a specific reference point.

## How It Works

The SQRBF activation function is defined as f(x) = exp(-ß * x²), where ß is a parameter that controls
the width of the Gaussian bell curve. This function outputs values between 0 and 1, with the maximum value
of 1 occurring when the input is 0, and values approaching 0 as the input moves away from 0 in either direction.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SQRBFActivation(Double)` | Initializes a new instance of the `SQRBFActivation` class with the specified beta parameter. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Activate()` | Applies the SQRBF activation function to a single input value. |
| `Activate(Tensor<>)` | Applies SQRBF to a tensor via engine primitives so the gradient tape records every step. |
| `AiDotNet#ActivationFunctions#Fused#IFusedActivation#TryGetFusedActivation(FusedActivationType)` |  |
| `ApplyToGraph(ComputationNode<>)` | Applies this activation function to a computation graph node. |
| `Derivative()` | Calculates the derivative of the SQRBF function for a given input value. |
| `SupportsScalarOperations` | Indicates whether this activation function supports scalar operations. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_beta` | The width parameter that controls the shape of the Gaussian bell curve. |

