---
title: "KDEBasedSelector<T>"
description: "Kernel Density Estimation (KDE) based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Density`

Kernel Density Estimation (KDE) based Feature Selection.

## For Beginners

KDE estimates the probability distribution of
your data smoothly. This selector finds features where different classes
have clearly different distributions, making it easier to tell classes apart.

## How It Works

Uses kernel density estimation to select features that maximize the
separation of class-conditional density estimates.

