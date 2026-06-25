---
title: "FisherExactTest<T>"
description: "Fisher's Exact Test for categorical feature selection with small samples."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Fisher's Exact Test for categorical feature selection with small samples.

## For Beginners

When you have categorical features and binary targets,
this test determines if there's a significant relationship. Unlike chi-square,
it works well even with small sample sizes or sparse cells.

## How It Works

Fisher's Exact Test computes the exact probability of observing the contingency
table assuming independence. Unlike chi-square, it's accurate for small samples.

