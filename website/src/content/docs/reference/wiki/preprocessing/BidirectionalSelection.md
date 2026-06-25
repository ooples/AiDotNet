---
title: "BidirectionalSelection<T>"
description: "Bidirectional (Stepwise) feature selection combining forward and backward steps."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Bidirectional (Stepwise) feature selection combining forward and backward steps.

## For Beginners

Pure forward selection can get stuck with a bad early
choice. Bidirectional selection can fix this by occasionally removing a feature
that's no longer useful after others were added. It's like being able to change
your mind as you build the feature set.

## How It Works

Alternates between forward selection (adding features) and backward elimination
(removing features). This allows correcting early poor choices and often finds
better subsets than pure forward or backward methods.

