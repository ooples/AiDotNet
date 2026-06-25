---
title: "TimeSeriesSplit"
description: "Provides time-series cross-validation with expanding or sliding window splits."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Preprocessing.TimeSeries`

Provides time-series cross-validation with expanding or sliding window splits.

## For Beginners

When working with time series data (like stock prices or weather),
you can't randomly split the data like you would for regular machine learning.

Why? Because predicting the future using future data is cheating! If you train on
data from 2023 to predict 2022, you're using information you wouldn't have had.

TimeSeriesSplit creates splits like this:

Each training set includes all previous data, and the test set is always "in the future"
relative to the training data.

## How It Works

Time series data requires special cross-validation that respects temporal ordering.
Unlike standard k-fold cross-validation, time series splits ensure that:

- Training data always comes before validation data
- No future information "leaks" into the training set

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `TimeSeriesSplit(Int32,Nullable<Int32>,Nullable<Int32>,Int32)` | Creates a new TimeSeriesSplit with the specified configuration. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Gap` | Gets the gap between train and test sets, useful to avoid data leakage. |
| `MaxTrainSize` | Gets the maximum training set size. |
| `NSplits` | Gets the number of splits to generate. |
| `TestSize` | Gets the test set size for each split. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CrossValidate([],Func<[],[],Double>)` | Performs cross-validation on time series data using the specified evaluation function. |
| `GetNSplits(Int32)` | Gets the number of splits that will be generated for the given number of samples. |
| `GetSplitSummary(Int32)` | Creates a summary of the split configuration for diagnostics. |
| `Split(Int32)` | Generates train/test index splits for the given number of samples. |

