---
title: "ExperimentInfo<T>"
description: "Contains structured experiment tracking information from a trained model."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models.Results`

Contains structured experiment tracking information from a trained model.

## For Beginners

This is a container for all experiment-related data.

It includes:

- Experiment and run IDs for finding this specific training session
- Access to the experiment tracker for comparing runs
- Training metrics history for visualization
- Hyperparameters used during training
- Data version information for reproducibility

## How It Works

This record provides type-safe access to experiment tracking data, including
the experiment and run identifiers, training metrics history, and hyperparameters.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExperimentInfo(String,String,IExperimentRun<>,IExperimentTracker<>,Dictionary<String,List<Double>>,Dictionary<String,Object>,Nullable<Int32>,String)` | Contains structured experiment tracking information from a trained model. |

