---
title: "HingeLossMetric<T>"
description: "Computes Hinge Loss: loss function used by Support Vector Machines (SVM)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Hinge Loss: loss function used by Support Vector Machines (SVM).

## For Beginners

Hinge loss penalizes predictions that are on the wrong side
of the decision boundary OR are correct but not confident enough.

- Loss = 0: All predictions are correct with margin ≥ 1
- Loss increases when predictions are wrong or uncertain

Used in SVM training and for evaluating margin-based classifiers.

## How It Works

Hinge Loss = (1/N) * Σmax(0, 1 - y_i * ŷ_i) where y ∈ {-1, +1}

**Important:** Predictions should be raw decision function values (continuous,
typically unbounded) from SVM, not class labels. Actuals should be class labels (0/1 or -1/+1).
If predictions are already binary class labels, use 0-1 loss instead.

