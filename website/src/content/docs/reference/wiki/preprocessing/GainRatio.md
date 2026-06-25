---
title: "GainRatio<T>"
description: "Gain Ratio for feature selection, normalizing Information Gain by feature entropy."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter`

Gain Ratio for feature selection, normalizing Information Gain by feature entropy.

## For Beginners

Information Gain tends to favor features with many
unique values (like IDs). Gain Ratio fixes this by accounting for how complex
the feature is. A feature that tells you a lot AND is simple gets a higher score
than one that's complex but only slightly informative.

## How It Works

Gain Ratio addresses Information Gain's bias toward features with many values
by dividing by the feature's intrinsic information (entropy of the feature itself).

