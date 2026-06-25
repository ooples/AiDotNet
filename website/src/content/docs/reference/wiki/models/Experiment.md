---
title: "Experiment"
description: "Represents a machine learning experiment that groups related training runs."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Models`

Represents a machine learning experiment that groups related training runs.

## How It Works

**For Beginners:** An experiment is a container for organizing related training runs.
It helps you group all attempts at solving a particular ML problem together.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `Experiment` | Private constructor for JSON deserialization. |
| `Experiment(String,String,Dictionary<String,String>)` | Initializes a new instance of the Experiment class. |

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
| `Touch` | Updates the last updated timestamp. |

