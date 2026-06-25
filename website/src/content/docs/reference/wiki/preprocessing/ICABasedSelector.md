---
title: "ICABasedSelector<T>"
description: "Independent Component Analysis (ICA) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Projection`

Independent Component Analysis (ICA) based Feature Selection.

## For Beginners

ICA finds hidden "sources" that combine to create
your observed data (like separating mixed audio signals). Features that
strongly contribute to these independent sources capture unique, non-redundant
information and are good candidates for selection.

## How It Works

Uses ICA to find independent components and selects features that
contribute most to these statistically independent sources.

