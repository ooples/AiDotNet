---
title: "TimeSeriesClassifierOptions<T>"
description: "Configuration options for time series classifiers."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for time series classifiers.

## For Beginners

These options configure how a time series classifier processes
sequence data. The most important parameters are the expected sequence length and number
of channels (variables) in your time series data.

## Properties

| Property | Summary |
|:-----|:--------|
| `NormalizeSequences` | Gets or sets whether to normalize sequences before processing. |
| `NumChannels` | Gets or sets the number of channels (variables) in the time series. |
| `SequenceLength` | Gets or sets the expected sequence length for input time series. |
| `SubsequenceLength` | Gets or sets the length of subsequences if `UseSubsequences` is true. |
| `UseSubsequences` | Gets or sets whether to use subsequence extraction for data augmentation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Validate` | Validates the options. |

