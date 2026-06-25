---
title: "UniformInitializationStrategy<T>"
description: "Uniform random initialization with configurable range."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Initialization`

Uniform random initialization with configurable range.

## For Beginners

This initializes weights with random values spread
evenly across a range. The default range [-0.05, 0.05] is a common starting point.

## How It Works

Simple uniform initialization in [-bound, bound]. Useful as a baseline
or when you need fine control over the initialization range.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `UniformInitializationStrategy(Double)` | Creates a uniform initialization strategy. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsLazy` |  |
| `LoadFromExternal` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `InitializeBiases(Tensor<>)` |  |
| `InitializeWeights(Tensor<>,Int32,Int32)` |  |

