---
title: "MCP<T>"
description: "Minimax Concave Penalty (MCP) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Minimax Concave Penalty (MCP) for feature selection.

## For Beginners

MCP is like LASSO but smarter about large coefficients.
LASSO shrinks everything, even important features. MCP says "if a coefficient is
big enough, stop penalizing it" - so truly important features keep their full effect
while unimportant ones are still shrunk to zero.

## How It Works

MCP is a non-convex penalty that provides an unbiased selection of significant
variables while controlling model complexity. Unlike LASSO, MCP does not over-penalize
large coefficients, providing nearly unbiased estimates for important features.

