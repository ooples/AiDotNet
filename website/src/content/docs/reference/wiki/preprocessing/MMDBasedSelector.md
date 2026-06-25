---
title: "MMDBasedSelector<T>"
description: "Maximum Mean Discrepancy (MMD) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Kernel`

Maximum Mean Discrepancy (MMD) based Feature Selection.

## For Beginners

MMD measures how different two probability distributions
are. By computing MMD between different classes, we can find features that make
the classes most distinguishable. Features with high MMD are good at separating
classes.

## How It Works

Uses MMD to measure the distribution difference between classes in kernel space,
selecting features that maximize this discrepancy.

