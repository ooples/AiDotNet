---
title: "WelchTTest<T>"
description: "Welch's t-test for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

Welch's t-test for feature selection.

## For Beginners

When comparing two groups that might have different
amounts of spread (variance), Welch's test is more reliable than the standard
t-test. It's generally recommended as the default choice for comparing two
group means.

## How It Works

Welch's t-test is a variant of the Student's t-test that does not assume equal
variances between the two groups. It uses the Welch-Satterthwaite approximation
for degrees of freedom.

