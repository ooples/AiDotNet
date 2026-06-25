---
title: "BidirectionalSearch<T>"
description: "Bidirectional Feature Search for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Wrapper`

Bidirectional Feature Search for feature selection.

## For Beginners

Instead of just adding features (forward) or just
removing them (backward), this method does both alternately. This can escape
local optima and find feature combinations that neither approach would find alone.

## How It Works

Bidirectional search combines forward selection and backward elimination in an
interleaved manner. It can add or remove features at each step, potentially
finding better solutions than either method alone.

