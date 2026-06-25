---
title: "MarkednessMetric<T>"
description: "Computes Markedness: PPV + NPV - 1 = Precision + NPV - 1."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Markedness: PPV + NPV - 1 = Precision + NPV - 1.

## For Beginners

Markedness measures how well the classifier "marks" the predictions:

- +1: Perfect classifier (PPV = NPV = 1)
- 0: Random classifier
- -1: Perfectly wrong classifier

Markedness is the dual of Informedness in the confusion matrix. While Informedness looks at
rows (actual classes), Markedness looks at columns (predicted classes).

## How It Works

Markedness = PPV + NPV - 1 = Precision + NPV - 1

