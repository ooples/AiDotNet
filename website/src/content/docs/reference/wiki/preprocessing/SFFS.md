---
title: "SFFS<T>"
description: "Sequential Floating Forward Selection (SFFS) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Sequential Floating Forward Selection (SFFS) for feature selection.

## For Beginners

Regular forward selection adds features one at a time
without looking back. SFFS is smarter - after adding a new feature, it reconsiders
whether any previously added features are now redundant. It's like building a team
and periodically reconsidering earlier hires as the team evolves.

## How It Works

SFFS is an advanced sequential search that combines forward selection with
conditional backward steps. After adding a feature, it tries removing previously
added features to see if removing them improves performance. This allows
correction of earlier suboptimal decisions.

