---
title: "RidgeFS<T>"
description: "Ridge Regression-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Ridge Regression-based feature selection.

## For Beginners

Ridge regression shrinks all feature weights
towards zero but never quite reaches it. Unlike LASSO, it keeps all features
but makes less important ones smaller. We select features by choosing those
with the biggest weights after shrinkage.

## How It Works

Ridge Regression uses L2 regularization which shrinks all coefficients but
doesn't set them to exactly zero. Feature selection is done by selecting
features with the largest absolute coefficients.

