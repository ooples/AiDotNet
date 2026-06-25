---
title: "WelchTTest<T>"
description: "Welch's t-test for binary classification with unequal variances."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate`

Welch's t-test for binary classification with unequal variances.

## For Beginners

Regular t-test assumes both classes have similar
spread (variance). If one class has much more variation than the other, Welch's
t-test gives more accurate results. It's generally safer to use when you're
unsure about variance equality.

## How It Works

Welch's t-test is an adaptation of Student's t-test that doesn't assume equal
variances between groups. It uses the Welch-Satterthwaite approximation for
degrees of freedom.

