---
title: "ElasticNetFeatureSelection<T>"
description: "Elastic Net regularization-based feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Elastic Net regularization-based feature selection.

## For Beginners

Elastic Net is a compromise between Lasso and Ridge
regression. When features are highly correlated, Lasso tends to pick one and ignore
the others. Elastic Net groups correlated features together, selecting them as a
group rather than arbitrarily picking one.

## How It Works

Elastic Net combines L1 (Lasso) and L2 (Ridge) regularization. Like Lasso, it can
drive coefficients to zero for feature selection, but the L2 component helps when
features are correlated. The l1_ratio parameter controls the mix between L1 and L2.

