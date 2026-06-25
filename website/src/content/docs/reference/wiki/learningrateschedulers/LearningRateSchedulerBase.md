---
title: "LearningRateSchedulerBase"
description: "Base class for learning rate schedulers providing common functionality."
section: "API Reference"
---

`Base Classes` · `AiDotNet.LearningRateSchedulers`

Base class for learning rate schedulers providing common functionality.

## For Beginners

This is the foundation that all learning rate schedulers build upon.
It handles the common tasks like keeping track of what step we're on and saving/loading state
so that training can be resumed from a checkpoint.

## How It Works

This abstract base class implements the common behavior for all learning rate schedulers,
including state management, step tracking, and serialization support.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `LearningRateSchedulerBase(Double,Double)` | Initializes a new instance of the LearningRateSchedulerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `BaseLearningRate` |  |
| `CurrentLearningRate` |  |
| `CurrentStep` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ComputeLearningRate(Int32)` | Computes the learning rate for a given step. |
| `GetLearningRateAtStep(Int32)` |  |
| `GetState` |  |
| `LoadState(Dictionary<String,Object>)` |  |
| `Reset` |  |
| `Step` |  |

## Fields

| Field | Summary |
|:-----|:--------|
| `_baseLearningRate` | The base (initial) learning rate. |
| `_currentLearningRate` | The current learning rate. |
| `_currentStep` | The current step count. |
| `_minLearningRate` | The minimum learning rate (floor). |

