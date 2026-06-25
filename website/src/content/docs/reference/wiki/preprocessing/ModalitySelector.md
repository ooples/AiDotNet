---
title: "ModalitySelector<T>"
description: "Modality based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Distribution`

Modality based Feature Selection.

## For Beginners

Modality refers to how many peaks a distribution has.
Unimodal = 1 peak, bimodal = 2 peaks, multimodal = many peaks. Features with
multiple modes often indicate natural groupings or distinct populations in the data.

## How It Works

Selects features based on the estimated number of modes (peaks) in their
distributions, using a histogram-based peak detection approach.

