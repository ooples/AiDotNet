---
title: "ModelRegistry"
description: "Local file-based model registry for managing model versions and deployments."
section: "API Reference"
---

`Models & Types` · `AiDotNet.TrainingMonitoring.ExperimentTracking`

Local file-based model registry for managing model versions and deployments.

## How It Works

**For Beginners:** ModelRegistry provides a centralized place to store,
version, and manage your trained models. It tracks:

- Different versions of your models
- Which version is in production
- How models were created (lineage)
- Deployment history

Example usage:

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelRegistry(String)` | Creates a new model registry. |

## Properties

| Property | Summary |
|:-----|:--------|
| `RegistryUri` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateModelVersion(String,String,String,String,Dictionary<String,String>)` |  |
| `CreateRegisteredModel(String,String,Dictionary<String,String>)` |  |
| `DeleteModelVersion(String,Int32)` |  |
| `DeleteModelVersionTag(String,Int32,String)` |  |
| `DeleteRegisteredModel(String)` |  |
| `Dispose` | Disposes the registry. |
| `GetCurrentDeployment(String)` |  |
| `GetDeploymentHistory(String,Int32)` |  |
| `GetLatestVersion(String,ModelStage[])` |  |
| `GetModelLineage(String,Int32)` |  |
| `GetModelVersion(String,Int32)` |  |
| `GetRegisteredModel(String)` |  |
| `ListModelVersions(String,ModelStage[])` |  |
| `ListRegisteredModels(String,String,Int32)` |  |
| `LoadModelArtifacts(String)` |  |
| `RecordDeployment(String,Int32,DeploymentInfo)` |  |
| `RecordModelLineage(String,Int32,ModelLineage)` |  |
| `SearchModelVersions(String,Int32)` |  |
| `SetModelVersionTag(String,Int32,String,String)` |  |
| `TransitionModelVersionStage(String,Int32,ModelStage,Boolean)` |  |
| `UpdateModelVersion(String,Int32,String)` |  |
| `UpdateRegisteredModel(String,String)` |  |

