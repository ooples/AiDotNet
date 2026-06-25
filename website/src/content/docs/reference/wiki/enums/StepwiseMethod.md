---
title: "StepwiseMethod"
description: "Specifies the direction of feature selection in stepwise regression and other statistical models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Specifies the direction of feature selection in stepwise regression and other statistical models.

## How It Works

**For Beginners:** Stepwise methods help AI decide which information is important to consider.

Imagine you're trying to predict house prices. There are many factors that could affect the price:
square footage, number of bedrooms, location, age of the house, etc. But using too many factors
can make your model complicated and less accurate.

Stepwise methods help you decide which factors (called "features" in AI) to include in your model.
They work by either starting with nothing and adding important features one by one, or starting
with everything and removing less important features one by one.

Think of it like packing for a trip: you can either start with an empty suitcase and add only what
you need (Forward), or start with everything you own and remove what you don't need (Backward).

## Fields

| Field | Summary |
|:-----|:--------|
| `Backward` | Starts with all features and removes them one at a time based on their lack of statistical significance. |
| `Forward` | Starts with no features and adds them one at a time based on their statistical significance. |

