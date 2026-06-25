---
title: "HellingerKernel<T>"
description: "Implements the Hellinger kernel for measuring similarity between probability distributions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Kernels`

Implements the Hellinger kernel for measuring similarity between probability distributions.

## For Beginners

A kernel function is a mathematical tool that measures how similar two data points are.
The Hellinger kernel is specifically designed for comparing data that represents probabilities or
frequencies (like how often words appear in documents, or the distribution of pixel intensities in images).

## How It Works

The Hellinger kernel is based on the Hellinger distance, which is a metric used to quantify the similarity
between two probability distributions. This kernel is particularly useful when working with data that
represents distributions, such as histograms, word frequencies, or any normalized non-negative data.

Think of this kernel as a "distribution comparator" that works well when your data points represent
how things are distributed or how frequently they occur. It's particularly good at handling sparse data
(where many values are zero) and is less sensitive to outliers than some other kernels.

This kernel works best when your data contains non-negative values that sum to 1 (or can be normalized
to sum to 1), such as probability distributions, normalized histograms, or frequency counts.

Note: For proper use of the Hellinger kernel, input vectors should contain non-negative values and
ideally should be normalized (sum to 1). If your data doesn't meet these criteria, you might need
to preprocess it before using this kernel.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `HellingerKernel` | Initializes a new instance of the Hellinger kernel. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Calculate(Vector<>,Vector<>)` | Calculates the Hellinger kernel value between two vectors. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |

