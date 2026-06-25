---
title: "StudentTTest<T>"
description: "Student's t-test for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical`

Student's t-test for feature selection.

## For Beginners

This test asks: "Is the average value of this feature
different enough between the two classes that it's probably not due to random
chance?" Features with large differences are more likely to be useful for
distinguishing between classes.

## How It Works

The Student's t-test compares the means of two groups to determine if they are
statistically different. For feature selection, it tests whether each feature
has significantly different values between positive and negative classes.

