---
title: "ElasticNetSelector<T>"
description: "Elastic Net (L1+L2) regularization-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Elastic Net (L1+L2) regularization-based feature selection.

## For Beginners

Elastic Net is like having both a strict budget (L1)
and a preference for simpler solutions (L2). This combination works well when
features are correlated - pure Lasso might arbitrarily pick one from a group
of correlated features, but Elastic Net tends to select or exclude them together.

## How It Works

Combines L1 (Lasso) and L2 (Ridge) penalties to perform feature selection
while handling correlated features better than pure Lasso. The l1_ratio
parameter controls the mix of L1 and L2 penalties.

