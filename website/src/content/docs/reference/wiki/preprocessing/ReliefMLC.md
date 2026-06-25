---
title: "ReliefMLC<T>"
description: "Relief for Multi-Label Classification (ReliefMLC)."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.MultiLabel`

Relief for Multi-Label Classification (ReliefMLC).

## For Beginners

Standard Relief compares single labels to find
hits/misses. ReliefMLC compares label sets, accounting for partial overlap
in multi-label scenarios.

## How It Works

Extends the Relief algorithm to handle multi-label problems. Uses Hamming
distance on label sets to determine similarity between instances.

