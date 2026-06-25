---
title: "IExperiment"
description: "Represents a machine learning experiment that groups related training runs."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Represents a machine learning experiment that groups related training runs.

## How It Works

**For Beginners:** An experiment is a container for organizing related training runs.
Think of it like a folder that groups all the attempts you make at solving a particular
machine learning problem. For example, you might have an experiment called "Customer Churn Prediction"
that contains all your different attempts at building a churn prediction model.

## Properties

| Property | Summary |
|:-----|:--------|
| `CreatedAt` | Gets the timestamp when the experiment was created. |
| `Description` | Gets or sets the description of the experiment. |
| `ExperimentId` | Gets the unique identifier for this experiment. |
| `LastUpdatedAt` | Gets the timestamp of the last update to the experiment. |
| `Name` | Gets or sets the name of the experiment. |
| `Status` | Gets the current status of the experiment. |
| `Tags` | Gets or sets tags associated with the experiment. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Archive` | Archives this experiment, making it read-only. |
| `Restore` | Restores this experiment from archived status. |

