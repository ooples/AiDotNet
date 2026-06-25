---
title: "FloatingBackwardSelector<T>"
description: "Sequential Floating Backward Selection (SFBS)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Sequential`

Sequential Floating Backward Selection (SFBS).

## For Beginners

This is the reverse of SFFS. It starts with all
features and removes them one by one, but can also add features back if doing
so improves the result. The "floating" helps avoid getting stuck with bad choices.

## How It Works

Starts with all features and combines backward elimination with conditional
forward steps to find better feature subsets.

