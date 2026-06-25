---
title: "ModelRegistry<T, TInput, TOutput>"
description: "Implementation of model registry for managing trained model storage and versioning."
section: "API Reference"
---

`Models & Types` · `AiDotNet.ModelRegistry`

Implementation of model registry for managing trained model storage and versioning.

## How It Works

**For Beginners:** This is a complete implementation of a model registry that manages
the lifecycle of your trained models.

Features include:

- Model versioning (track different versions of the same model)
- Lifecycle stages (Development, Staging, Production, Archived)
- Model comparison and lineage tracking
- Persistent storage with JSON serialization

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelRegistry(String)` | Initializes a new instance of the ModelRegistry class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `LoadErrors` | List of errors that occurred during model loading at startup. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArchiveModel(String,Int32)` | Archives a model version. |
| `AttachModelCard(String,Int32,ModelCard)` | Attaches a Model Card to a registered model version. |
| `CloneRegisteredModel(RegisteredModel<,,>)` | Creates a shallow clone of a registered model to prevent external modification. |
| `CompareModels(String,Int32,Int32)` | Compares two model versions. |
| `CreateModelVersion(String,IModel<,,>,ModelMetadata<>,String)` | Creates a new version of an existing model. |
| `DeleteModel(String)` | Deletes all versions of a model. |
| `DeleteModelVersion(String,Int32)` | Deletes a specific model version. |
| `GenerateModelCard(String,Int32,String)` | Generates a Model Card from the registered model's metadata and evaluation results. |
| `GetInternalModel(String,Int32)` | Gets the internal model for mutation operations. |
| `GetLatestModel(String)` | Gets the latest version of a model. |
| `GetModel(String,Nullable<Int32>)` | Retrieves a specific model version from the registry. |
| `GetModelByStage(String,ModelStage)` | Gets the model currently in a specific stage. |
| `GetModelCard(String,Int32)` | Gets the Model Card for a registered model version. |
| `GetModelLineage(String,Int32)` | Gets the lineage information for a model. |
| `GetModelStoragePath(String,Int32)` | Gets the storage location for a model version. |
| `ListModelVersions(String)` | Lists all versions of a specific model. |
| `ListModels(String,Dictionary<String,String>)` | Lists all models in the registry. |
| `RegisterModel(String,IModel<,,>,ModelMetadata<>,Dictionary<String,String>)` | Registers a new model in the registry. |
| `SaveModelCard(String,Int32,String)` | Saves the Model Card for a model version to a file. |
| `SearchModels(ModelSearchCriteria<>)` | Searches for models based on criteria. |
| `TransitionModelStage(String,Int32,ModelStage,Boolean)` | Transitions a model version to a different stage. |
| `UpdateModelMetadata(String,Int32,ModelMetadata<>)` | Updates the metadata for a model version. |
| `UpdateModelTags(String,Int32,Dictionary<String,String>)` | Adds or updates tags for a model. |

