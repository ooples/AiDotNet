---
title: "EmbeddedWrapper<T>"
description: "Embedded-Wrapper hybrid feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Hybrid`

Embedded-Wrapper hybrid feature selection.

## For Beginners

This is like having an expert advisor (embedded method)
suggest which features might be important, then carefully verifying those
suggestions (wrapper). The advisor speeds up the search while the verification
ensures accuracy.

## How It Works

Embedded-Wrapper combines embedded methods (like regularization) with wrapper
search. It uses embedded feature importance to guide the wrapper search,
making the search more efficient.

