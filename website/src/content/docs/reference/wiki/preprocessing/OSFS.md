---
title: "OSFS<T>"
description: "Online Streaming Feature Selection (OSFS) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Online`

Online Streaming Feature Selection (OSFS) for feature selection.

## For Beginners

Imagine features arriving one by one like a stream.
For each new feature, OSFS asks: "Does this add useful information beyond what
I already have?" If yes, it keeps the feature. It also checks if any existing
features become redundant after adding the new one.

## How It Works

OSFS performs feature selection in an online/streaming fashion where features
arrive one at a time. It maintains a set of selected features and uses
conditional independence tests to decide whether to add new features.

