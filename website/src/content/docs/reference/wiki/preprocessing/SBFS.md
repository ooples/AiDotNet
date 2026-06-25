---
title: "SBFS<T>"
description: "Sequential Backward Floating Selection (SBFS) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Sequential Backward Floating Selection (SBFS) for feature selection.

## For Beginners

SBFS starts with all features and removes them one
by one, but it's smart enough to add features back if removing them was a bad
idea. It's like cleaning out a closet - you might throw something away, then
realize you need it after throwing other things out.

## How It Works

SBFS is a wrapper method that starts with all features and iteratively removes
the least useful ones. Unlike simple backward elimination, SBFS can conditionally
add features back if their removal was a mistake given later removals.

