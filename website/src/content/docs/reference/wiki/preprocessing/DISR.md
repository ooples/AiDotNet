---
title: "DISR<T>"
description: "Double Input Symmetric Relevance (DISR) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory`

Double Input Symmetric Relevance (DISR) for feature selection.

## For Beginners

DISR looks at how pairs of features together
relate to the target, using a normalized measure that accounts for both
directions of information flow. This helps find features that complement
each other in predicting the target.

## How It Works

DISR extends mutual information by considering symmetric relevance between
pairs of features and the target. It uses the concept of joint symmetric
uncertainty to measure feature relevance.

