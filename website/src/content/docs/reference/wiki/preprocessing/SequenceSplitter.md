---
title: "SequenceSplitter<T>"
description: "Sequence splitter for sequential data like text, DNA, or user sessions."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.DataPreparation.Splitting.DomainSpecific`

Sequence splitter for sequential data like text, DNA, or user sessions.

## For Beginners

Sequential data has an inherent order that matters.
Examples include:

- Text documents (sequence of words)
- DNA sequences (sequence of nucleotides)
- User clickstreams (sequence of page visits)

## How It Works

**Split Strategies:**

- By sequence: Keep entire sequences together
- By position: Split within sequences at a certain position
- By time: For time-stamped sequences

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `SequenceSplitter(Double,Int32,Boolean,Int32)` | Creates a new sequence splitter. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Description` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `Split(Matrix<>,Vector<>)` |  |
| `WithSequenceIds(Int32[])` | Sets explicit sequence IDs for each sample. |

