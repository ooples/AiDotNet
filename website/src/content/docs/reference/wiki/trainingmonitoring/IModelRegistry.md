---
title: "IModelRegistry"
description: "Interface for a model registry that manages model versions and deployments."
section: "API Reference"
---

`Interfaces` · `AiDotNet.TrainingMonitoring.ExperimentTracking`

Interface for a model registry that manages model versions and deployments.

## How It Works

**For Beginners:** A Model Registry is like a central library for your
trained models. It helps you:

- Store and version models
- Track which model is in production
- Manage model lifecycle (staging -> production -> archived)
- Record model lineage (what data/experiments created this model)

Think of it like the MLflow Model Registry - it's where your best models
get promoted for production use.

Key concepts:

- Registered Model: A named model (e.g., "fraud-detector")
- Model Version: A specific version of that model (e.g., v1, v2)
- Stage: Where the model is in its lifecycle (Staging, Production, Archived)
- Lineage: The experiment/run/data that created this model

## Properties

| Property | Summary |
|:-----|:--------|
| `RegistryUri` | Gets the registry URI. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateModelVersion(String,String,String,String,Dictionary<String,String>)` | Creates a new model version. |
| `CreateRegisteredModel(String,String,Dictionary<String,String>)` | Registers a new model or gets existing. |
| `DeleteModelVersion(String,Int32)` | Deletes a model version. |
| `DeleteModelVersionTag(String,Int32,String)` | Deletes a tag from a model version. |
| `DeleteRegisteredModel(String)` | Deletes a registered model and all its versions. |
| `GetCurrentDeployment(String)` | Gets the current production deployment. |
| `GetDeploymentHistory(String,Int32)` | Gets deployment history for a model version. |
| `GetLatestVersion(String,ModelStage[])` | Gets the latest model version. |
| `GetModelLineage(String,Int32)` | Gets model lineage information. |
| `GetModelVersion(String,Int32)` | Gets a specific model version. |
| `GetRegisteredModel(String)` | Gets a registered model by name. |
| `ListModelVersions(String,ModelStage[])` | Lists all versions of a model. |
| `ListRegisteredModels(String,String,Int32)` | Lists all registered models. |
| `LoadModelArtifacts(String)` | Loads a model from the registry. |
| `RecordDeployment(String,Int32,DeploymentInfo)` | Records a deployment of a model version. |
| `RecordModelLineage(String,Int32,ModelLineage)` | Records model lineage information. |
| `SearchModelVersions(String,Int32)` | Searches for model versions. |
| `SetModelVersionTag(String,Int32,String,String)` | Sets a tag on a model version. |
| `TransitionModelVersionStage(String,Int32,ModelStage,Boolean)` | Transitions a model version to a new stage. |
| `UpdateModelVersion(String,Int32,String)` | Updates a model version. |
| `UpdateRegisteredModel(String,String)` | Updates a registered model. |

