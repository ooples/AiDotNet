---
title: "MCPSelector<T>"
description: "Minimax Concave Penalty (MCP) for feature selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Embedded`

Minimax Concave Penalty (MCP) for feature selection.

## For Beginners

Like SCAD, MCP is an improvement over Lasso.
It strongly penalizes small coefficients (pushing them to zero) but
gradually reduces the penalty for larger coefficients. This means truly
important features keep their full effect while noise features are eliminated.

## How It Works

MCP is a nonconvex penalty similar to SCAD that provides nearly unbiased
estimates for large coefficients while inducing sparsity. It has a concave
shape that reduces shrinkage bias.

