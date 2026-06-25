---
title: "SampleEntropySelector<T>"
description: "Sample Entropy (SampEn) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Entropy`

Sample Entropy (SampEn) based Feature Selection.

## For Beginners

Sample entropy is like approximate entropy but
doesn't count self-matches, making it more reliable for shorter sequences.
It measures unpredictability - low SampEn means regular, predictable patterns;
high SampEn means complex, chaotic behavior.

## How It Works

Selects features based on their sample entropy, an improved version of
approximate entropy that is less biased for short time series.

