---
title: "ModifiedGradientDescentOptimizer"
description: "Modified Gradient Descent optimizer for Hope architecture."
section: "Reference"
---

_Optimizers_

Modified Gradient Descent optimizer for Hope architecture. Based on Equations 27-29 from "Nested Learning" paper. Traditional GD: W_{t+1} = W_t - η * ∇L(W_t; x_t) ⊗ x_t Modified GD: W_{t+1} = W_t * (I - x_t*x_t^T) - η * ∇L(W_t; x_t) ⊗ x_t This formulation uses L2 regression objective instead of dot-product similarity, resulting in better handling of data dependencies in token space.

