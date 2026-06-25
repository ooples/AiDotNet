---
title: "VotingFeatureSelector<T>"
description: "Voting-based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Ensemble`

Voting-based Feature Selection.

## For Beginners

Instead of trusting just one method to pick
features, this approach asks multiple methods to vote. Features that are
chosen by many different methods are more likely to be truly important,
giving you more confident selections.

## How It Works

Combines multiple feature selection methods using voting. Each method casts
votes for its top features, and features with the most votes are selected.

