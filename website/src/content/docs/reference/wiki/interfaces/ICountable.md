---
title: "ICountable"
description: "Defines capability to report dataset size and iteration progress."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines capability to report dataset size and iteration progress.

## For Beginners

Knowing how much data you have and where you are
in processing is essential for:

- Showing progress bars
- Knowing when an epoch (full pass through data) is complete
- Calculating metrics like "samples processed per second"

## How It Works

Data loaders that implement this interface provide information about
their size and current position, useful for progress tracking and
determining when an epoch is complete.

## Properties

| Property | Summary |
|:-----|:--------|
| `BatchCount` | Gets the total number of batches based on current batch size. |
| `CurrentBatchIndex` | Gets the current batch index in the iteration (0-based). |
| `CurrentIndex` | Gets the current sample index in the iteration (0-based). |
| `Progress` | Gets the progress through the current epoch as a value from 0.0 to 1.0. |
| `TotalCount` | Gets the total number of samples in the dataset. |

