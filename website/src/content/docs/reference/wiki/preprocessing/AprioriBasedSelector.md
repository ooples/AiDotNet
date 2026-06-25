---
title: "AprioriBasedSelector<T>"
description: "Apriori-based Association Rule Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Association`

Apriori-based Association Rule Feature Selection.

## For Beginners

Association rules find patterns like "people who
buy bread often buy butter." We apply this to features: features that
"associate" strongly with the target class (high confidence) and appear
often (high support) are selected.

## How It Works

Uses association rule mining concepts to find features that frequently
co-occur with the target class, selecting features with high support
and confidence.

