---
title: "NBEATSBlock"
description: "Represents a single block in the N-BEATS architecture."
section: "Reference"
---

_Time-Series Models_

Represents a single block in the N-BEATS architecture.

## For Beginners

A block is the basic building unit of N-BEATS. Think of it like a specialized predictor that: 1. Looks at the input time series 2. Tries to reconstruct what it saw (backcast) 3. Predicts the future (forecast) 4. Passes the "leftover" patterns it couldn't explain to the next block Multiple blocks work together, with each one focusing on different aspects of the data.

## How It Works

Each N-BEATS block consists of: 1. A stack of fully connected layers (the "theta" network) 2. A basis expansion layer for generating backcast (reconstruction of input) 3. A basis expansion layer for generating forecast (prediction of future) 

The block architecture implements a doubly residual stacking principle: - Backcast residual: Input minus backcast is passed to the next block - Forecast addition: Forecasts from all blocks are summed for the final prediction

