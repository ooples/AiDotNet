---
title: "TransferLearningSelector<T>"
description: "Transfer Learning-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Transfer`

Transfer Learning-based Feature Selection.

## For Beginners

Sometimes we have lots of data from one situation
(like medical records from one hospital) but little data from another
(a new hospital). Transfer learning uses what we learned about important
features from the first situation to help with the second.

## How It Works

Uses feature importance learned from a source domain to guide feature
selection in a target domain. Particularly useful when the target domain
has limited data but the source domain has abundant labeled data.

