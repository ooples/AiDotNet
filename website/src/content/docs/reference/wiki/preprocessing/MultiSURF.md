---
title: "MultiSURF<T>"
description: "MultiSURF algorithm with adaptive distance thresholds per instance."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Relief`

MultiSURF algorithm with adaptive distance thresholds per instance.

## For Beginners

While SURF uses one global distance threshold,
MultiSURF calculates a custom threshold for each sample based on how
spread out its neighbors are. This works better for datasets with
varying density regions.

## How It Works

MultiSURF improves SURF by using instance-specific distance thresholds
based on the local density around each instance. This adapts to varying
data densities across the feature space.

