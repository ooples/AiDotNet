---
title: "EnsembleFitDetectorOptions"
description: "Configuration options for the Ensemble Fit Detector, which combines multiple model fitness detectors to provide more robust and accurate recommendations for algorithm selection."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for the Ensemble Fit Detector, which combines multiple model fitness detectors
to provide more robust and accurate recommendations for algorithm selection.

## For Beginners

Think of this as getting advice from a panel of experts instead of just one person.
Each expert (detector) specializes in recognizing different patterns in your data. By combining their opinions,
you get more reliable recommendations about which AI algorithms will work best for your specific problem.
It's like asking several doctors for a diagnosis instead of relying on just one opinion.

## How It Works

An ensemble fit detector evaluates how well different algorithms might perform on a given dataset
by combining the opinions of multiple specialized detectors. Each detector in the ensemble focuses on
different aspects of the data and problem characteristics.

## Properties

| Property | Summary |
|:-----|:--------|
| `DetectorWeights` | Gets or sets the weights applied to each detector in the ensemble. |
| `MaxRecommendations` | Gets or sets the maximum number of algorithm recommendations to return. |

