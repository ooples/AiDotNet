---
title: "CohensKappaMetric<T>"
description: "Computes Cohen's Kappa: agreement measure that accounts for chance agreement."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Evaluation.Metrics.Classification`

Computes Cohen's Kappa: agreement measure that accounts for chance agreement.

## For Beginners

Kappa measures how much better the model is than random guessing.

- Kappa = 1: Perfect agreement
- Kappa = 0: No better than chance
- Kappa < 0: Worse than chance

## How It Works

Kappa = (p_o - p_e) / (1 - p_e)
where p_o = observed agreement, p_e = expected agreement by chance

**Interpretation guidelines (Landis & Koch):**
<0: Poor, 0-0.2: Slight, 0.2-0.4: Fair, 0.4-0.6: Moderate, 0.6-0.8: Substantial, 0.8-1: Almost perfect

