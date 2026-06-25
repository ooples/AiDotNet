---
title: "MCFS<T>"
description: "Multi-Cluster Feature Selection using spectral graph analysis."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral`

Multi-Cluster Feature Selection using spectral graph analysis.

## For Beginners

MCFS finds features that help preserve the natural
groupings (clusters) in your data. It first discovers hidden cluster structures
using spectral methods, then picks features that best capture those clusters.
Great when you don't have labels but want features that preserve data structure.

## How It Works

MCFS uses spectral analysis to find features that preserve the multi-cluster
structure in data. It computes cluster indicators via spectral clustering
and then selects features using sparse regression with L1 regularization.

