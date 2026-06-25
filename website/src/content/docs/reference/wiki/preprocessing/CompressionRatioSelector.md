---
title: "CompressionRatioSelector<T>"
description: "Compression Ratio based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Compression`

Compression Ratio based Feature Selection.

## For Beginners

If a feature's values can be described very briefly
(highly compressible), it probably doesn't contain much information. This selector
keeps features that are harder to compress because they have more variety.

## How It Works

Selects features based on their compressibility - features with higher entropy
(less compressible) often contain more useful information.

