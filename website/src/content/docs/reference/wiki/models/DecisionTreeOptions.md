---
title: "DecisionTreeOptions"
description: "Configuration options for decision tree algorithms."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for decision tree algorithms.

## For Beginners

A decision tree works like a flowchart that asks a series of questions about your data
to arrive at a prediction. Imagine playing a game of "20 Questions" where each question narrows down the possible answers.
These settings control how detailed the questions can get, how many questions to ask, and how to decide which
questions are most important.

## How It Works

Decision trees are machine learning models that make predictions by following a tree-like structure of decisions.
These options control how the decision tree is built and how it makes predictions.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxDepth` | Gets or sets the maximum depth (number of levels) of the decision tree. |
| `MaxFeatures` | Gets or sets the fraction of features to consider when looking for the best split. |
| `MinSamplesLeaf` | Gets or sets the minimum number of samples required in each leaf node after a split. |
| `MinSamplesSplit` | Gets or sets the minimum number of samples required to split an internal node. |
| `SoftTreeTemperature` | Gets or sets the temperature parameter for soft decision tree mode. |
| `SplitCriterion` | Gets or sets the criterion used to evaluate the quality of a split. |
| `UseSoftTree` | Gets or sets whether to use soft (differentiable) tree mode for JIT compilation support. |

