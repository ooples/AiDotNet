---
title: "McNemarTest<T>"
description: "McNemar's test: compares the performance of two binary classifiers on the same dataset."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Statistics`

McNemar's test: compares the performance of two binary classifiers on the same dataset.

## For Beginners

McNemar's test is specifically designed for comparing two
classifiers' predictions on the same dataset. It uses a 2x2 contingency table:

The test focuses on the disagreement cells (b and c) - cases where one model is right
and the other is wrong.

## How It Works

**When to use:**

- Comparing two classifiers on the same test set
- Binary classification problems
- When you want to test if classifier A is significantly better than B

## Methods

| Method | Summary |
|:-----|:--------|
| `Test([],[],Double)` | Performs McNemar's test using binary predictions. |

