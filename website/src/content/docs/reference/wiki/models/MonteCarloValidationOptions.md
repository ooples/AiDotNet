---
title: "MonteCarloValidationOptions"
description: "Represents the options for Monte Carlo cross-validation."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Represents the options for Monte Carlo cross-validation.

## For Beginners

Monte Carlo cross-validation options help you customize how the Monte Carlo method splits and tests your data.

What this class does:

- Inherits all the basic cross-validation options (like number of folds)
- Adds a new option to set the size of the validation set

This is useful because:

- It allows you to control how much of your data is used for validation in each Monte Carlo iteration
- You can adjust this to find the right balance between your training and validation set sizes

Think of it like deciding how to split a deck of cards for a card game - this option lets you choose 
how many cards go into each pile (training and validation) for each round of Monte Carlo testing.

## How It Works

This class extends the base CrossValidationOptions with additional properties specific to Monte Carlo cross-validation.

## Properties

| Property | Summary |
|:-----|:--------|
| `ValidationSize` | Gets or sets the size of the validation set as a proportion of the total dataset. |

