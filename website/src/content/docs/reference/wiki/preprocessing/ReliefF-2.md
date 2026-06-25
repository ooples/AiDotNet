---
title: "ReliefF<T>"
description: "ReliefF algorithm for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Relief`

ReliefF algorithm for feature selection.

## For Beginners

ReliefF works by looking at each data point and
comparing it to its nearest neighbors. Features are scored based on how well
they separate a point from neighbors of different classes while keeping it
close to neighbors of the same class. It's like finding features that cluster
similar items together.

## How It Works

ReliefF is an extension of the Relief algorithm that handles multi-class
problems and uses k nearest neighbors. It evaluates features based on their
ability to distinguish between instances of different classes.

