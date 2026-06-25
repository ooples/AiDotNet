---
title: "ICAP<T>"
description: "Interaction Capping (ICAP) feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Interaction Capping (ICAP) feature selection.

## For Beginners

ICAP looks for features that work well together.
Sometimes two features alone aren't very informative, but combined they reveal
patterns. ICAP tries to find these synergistic combinations while still avoiding
redundancy. It's like finding teammates who complement each other's abilities.

## How It Works

ICAP extends mRMR by adding interaction information. It considers not just pairwise
redundancy but also synergistic effects where combining features provides more
information than the sum of individual contributions.

