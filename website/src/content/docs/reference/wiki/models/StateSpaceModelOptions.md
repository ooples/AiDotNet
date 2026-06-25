---
title: "StateSpaceModelOptions<T>"
description: "Configuration options for State Space Models, which represent time series data through hidden states and observable outputs for forecasting and analysis."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for State Space Models, which represent time series data through
hidden states and observable outputs for forecasting and analysis.

## For Beginners

State Space Models help analyze time series data by tracking hidden variables that influence observable measurements.

In many real-world systems:

- We can only measure certain outputs (like temperature, price, or position)
- But these measurements are influenced by hidden internal states
- State Space Models help us track these hidden states over time

For example, in tracking a moving object:

- We might only observe its position at certain times (observations)
- But its velocity and acceleration are hidden states that affect future positions
- A state space model can estimate these hidden states from the observations

These models are powerful because they:

- Handle noisy measurements
- Can incorporate multiple influencing factors
- Provide a framework for forecasting future values
- Work well with missing data

Common applications include:

- Economic forecasting
- Object tracking in computer vision
- Signal processing
- Financial time series analysis

This class lets you configure the structure and training process for state space models.

## How It Works

State Space Models (SSMs) are a flexible class of time series models that represent the dynamics of a system 
through hidden states and observable outputs. They provide a unified framework for modeling various time series 
patterns, including trends, seasonality, and cycles. The core of a state space model consists of two equations: 
a state equation that describes how the hidden state evolves over time, and an observation equation that relates 
the hidden state to the observed data. Common examples of state space models include the Kalman Filter, 
Hidden Markov Models, and structural time series models. This class provides configuration options for state 
space models, including the dimensions of the state and observation vectors, learning parameters for model 
estimation, and convergence criteria. These options allow customization of the model complexity and fitting 
process to match the specific characteristics of the time series being analyzed.

## Properties

| Property | Summary |
|:-----|:--------|
| `LearningRate` | Gets or sets the learning rate for gradient-based parameter estimation. |
| `MaxIterations` | Gets or sets the maximum number of iterations for the parameter estimation algorithm. |
| `ObservationSize` | Gets or sets the dimension of the observation vector in the state space model. |
| `StateSize` | Gets or sets the dimension of the state vector in the state space model. |
| `Tolerance` | Gets or sets the convergence tolerance for the parameter estimation algorithm. |

