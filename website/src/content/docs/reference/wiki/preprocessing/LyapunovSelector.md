---
title: "LyapunovSelector<T>"
description: "Lyapunov Exponent based Feature Selection."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.FeatureSelection.Complexity`

Lyapunov Exponent based Feature Selection.

## For Beginners

The Lyapunov exponent measures how fast nearby
trajectories diverge over time. Positive values indicate chaos (butterfly effect),
zero indicates marginal stability, negative indicates convergence. Features with
positive Lyapunov exponents may contain complex nonlinear dynamics.

## How It Works

Selects features based on their largest Lyapunov exponent, which measures
sensitivity to initial conditions (chaos) in dynamical systems.

