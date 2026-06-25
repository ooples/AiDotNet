---
title: "FCI_Selector<T>"
description: "Fast Causal Inference (FCI) Algorithm Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Causal`

Fast Causal Inference (FCI) Algorithm Feature Selection.

## For Beginners

FCI extends the PC algorithm to handle hidden
(unmeasured) variables. It builds a partial ancestral graph and selects
features that have causal connections to the target, even accounting for
variables we can't observe.

## How It Works

Uses the FCI algorithm for causal discovery with latent confounders,
selecting features that are direct causes or effects of the target.

