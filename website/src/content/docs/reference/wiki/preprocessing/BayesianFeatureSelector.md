---
title: "BayesianFeatureSelector<T>"
description: "Bayesian Feature Selection using posterior probability."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Bayesian`

Bayesian Feature Selection using posterior probability.

## For Beginners

Bayesian methods treat feature relevance as a
probability question. Instead of saying "feature X is important" or "not important",
we say "there's a 90% chance feature X is important." This gives us uncertainty
estimates and allows us to incorporate prior knowledge about which features
might be relevant.

## How It Works

Uses Bayesian inference to compute the posterior probability that each
feature is relevant to the target. Features with high posterior probability
are selected.

