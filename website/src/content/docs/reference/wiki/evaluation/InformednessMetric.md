---
title: "InformednessMetric<T>"
description: "Computes Informedness (Youden's J statistic): TPR + TNR - 1 = Recall + Specificity - 1."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Informedness (Youden's J statistic): TPR + TNR - 1 = Recall + Specificity - 1.

## For Beginners

Informedness measures how "informed" the classifier is:

- +1: Perfect classifier
- 0: Random classifier (no better than chance)
- -1: Perfectly wrong classifier

Also known as Youden's J statistic. It's the optimal threshold point on a ROC curve and
considers both classes equally important.

## How It Works

Informedness = TPR + TNR - 1 = Sensitivity + Specificity - 1

