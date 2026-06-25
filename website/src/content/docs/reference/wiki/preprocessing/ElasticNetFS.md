---
title: "ElasticNetFS<T>"
description: "Elastic Net-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Elastic Net-based feature selection.

## For Beginners

Elastic Net is like LASSO but smarter about groups
of related features. Instead of picking just one from a correlated group (like
LASSO does), it can keep several if they're all useful. This is great when you
have features that measure similar things.

## How It Works

Elastic Net combines L1 (LASSO) and L2 (Ridge) regularization to select
features while handling correlated features better than pure LASSO.
Features with non-zero coefficients are selected.

