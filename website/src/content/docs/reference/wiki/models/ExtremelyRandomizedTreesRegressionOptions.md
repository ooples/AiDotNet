---
title: "ExtremelyRandomizedTreesRegressionOptions"
description: "Configuration options for Extremely Randomized Trees regression, an ensemble learning method that builds multiple decision trees with additional randomization for improved prediction accuracy."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Extremely Randomized Trees regression, an ensemble learning method
that builds multiple decision trees with additional randomization for improved prediction accuracy.

## For Beginners

Extremely Randomized Trees is like having a large committee of decision-makers
(trees) who each look at your data in a slightly different, randomized way. Imagine asking 100 people to
help you decide whether to buy a house, but each person can only consider a random subset of factors
(like price, location, size) and must make quick decisions without overthinking. By averaging all their
opinions, you often get better advice than from just one person or from a committee that all thinks the
same way. This randomness helps the model avoid focusing too much on specific patterns that might just be
coincidences in your training data.

## How It Works

Extremely Randomized Trees (also known as Extra Trees) is an ensemble learning method that extends
Random Forests by introducing additional randomization in the way splits are computed. While Random Forests
search for the optimal split among a random subset of features, Extra Trees select random splits for each
feature and choose the best among those. This additional randomization helps reduce variance and often
improves generalization.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDegreeOfParallelism` | Gets or sets the maximum number of trees that can be trained simultaneously. |
| `NumberOfTrees` | Gets or sets the number of decision trees to build in the ensemble. |

