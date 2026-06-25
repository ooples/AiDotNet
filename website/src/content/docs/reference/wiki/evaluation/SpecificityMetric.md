---
title: "SpecificityMetric<T>"
description: "Computes specificity (true negative rate): the proportion of actual negatives correctly identified."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes specificity (true negative rate): the proportion of actual negatives correctly identified.

## For Beginners

Specificity answers: "Of all actual negatives, how many did the model correctly identify?"
High specificity means few false positives. A spam filter with 99% specificity means it correctly
lets through 99% of legitimate emails.

## How It Works

Specificity = TN / (TN + FP) = True Negatives / Actual Negatives

