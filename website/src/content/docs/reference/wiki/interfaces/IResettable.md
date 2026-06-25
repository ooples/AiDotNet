---
title: "IResettable"
description: "Defines capability to reset iteration state back to the beginning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines capability to reset iteration state back to the beginning.

## For Beginners

Think of this like rewinding a video back to the start.
After processing all your data once (one epoch), you can reset and go through it again
for another epoch of training.

## How It Works

Data loaders that implement this interface can be reset to their initial state,
allowing iteration to start over from the beginning of the dataset.

## Methods

| Method | Summary |
|:-----|:--------|
| `Reset` | Resets the iteration state to the beginning of the data. |

