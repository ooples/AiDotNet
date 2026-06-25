---
title: "PairedTTest<T>"
description: "Paired t-test: compares means of two related samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

Paired t-test: compares means of two related samples.

## For Beginners

Use this test when you have paired observations, such as:

- Before/after measurements on the same subjects
- Same dataset evaluated by two different models (paired by sample)
- Cross-validation fold results for two models on the same folds

The test determines if there's a significant difference between the pairs.

## How It Works

**Assumptions:**

- The differences between pairs are approximately normally distributed
- Pairs are independent of each other
- Data is continuous (or at least ordinal with many levels)

