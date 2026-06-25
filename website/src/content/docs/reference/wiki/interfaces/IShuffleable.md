---
title: "IShuffleable"
description: "Defines capability to shuffle data for randomized iteration."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines capability to shuffle data for randomized iteration.

## For Beginners

Shuffling is like shuffling a deck of cards before dealing.
When training, you don't want your model to learn "cat images always come first,
then dog images" - you want it to learn actual features. Shuffling ensures each
epoch sees data in a different order.

## How It Works

Data loaders that implement this interface can shuffle their data,
which is important for training to prevent the model from learning
the order of examples rather than the patterns in the data.

## Properties

| Property | Summary |
|:-----|:--------|
| `IsShuffled` | Gets whether the data is currently shuffled. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Shuffle(Nullable<Int32>)` | Shuffles the data order using the specified seed for reproducibility. |
| `Unshuffle` | Restores the original (unshuffled) data order. |

