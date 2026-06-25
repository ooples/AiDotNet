---
title: "TrainingNotification"
description: "Represents a training notification."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.Notifications`

Represents a training notification.

## Properties

| Property | Summary |
|:-----|:--------|
| `CreatedAt` | Gets or sets when the notification was created. |
| `ExperimentName` | Gets or sets the experiment name. |
| `Message` | Gets or sets the notification message. |
| `Metadata` | Gets or sets additional metadata. |
| `RunId` | Gets or sets the run ID. |
| `Severity` | Gets or sets the notification severity. |
| `Title` | Gets or sets the notification title. |
| `Type` | Gets or sets the notification type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CheckpointSaved(String,String,Int32,String)` | Creates a checkpoint saved notification. |
| `EarlyStopping(String,Int32,Int32,Double,String,String)` | Creates an early stopping notification. |
| `HyperparameterProgress(String,Int32,Int32,Double,Dictionary<String,Object>,String)` | Creates a hyperparameter progress notification. |
| `NewBestModel(String,Int32,Double,String,String)` | Creates a new best model notification. |
| `ResourceWarning(String,Double,Double,String,String)` | Creates a resource warning notification. |
| `TrainingCompleted(String,Int32,Double,Nullable<Double>,Nullable<TimeSpan>,String)` | Creates a training completed notification. |
| `TrainingFailed(String,String,Nullable<Int32>,String)` | Creates a training failed notification. |
| `TrainingStarted(String,String,Dictionary<String,Object>)` | Creates a training started notification. |

