---
title: "MRMR<T>"
description: "Minimum Redundancy Maximum Relevance (mRMR) feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate`

Minimum Redundancy Maximum Relevance (mRMR) feature selection.

## For Beginners

mRMR tries to pick features that are both useful
(related to what you're predicting) and diverse (not repeating the same information).
It's like assembling a team - you want skilled people, but also people with
different skills rather than everyone being good at the same thing.

## How It Works

mRMR selects features that have maximum relevance to the target while having
minimum redundancy among themselves. It balances finding informative features
with avoiding duplicated information.

