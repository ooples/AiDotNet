---
title: "IModelRegistry<T, TInput, TOutput>"
description: "Defines the contract for model registry systems that manage trained model storage and versioning."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Interfaces`

Defines the contract for model registry systems that manage trained model storage and versioning.

## How It Works

A model registry serves as a centralized repository for trained models, managing their lifecycle
from development through production deployment.

**For Beginners:** Think of a model registry like a library for your trained models.
Just like a library catalogs books and tracks which are checked out, a model registry:

- Stores all your trained models in one place
- Tracks different versions of the same model
- Records which models are being used in production
- Helps you find and retrieve the right model when you need it

Key features include:

- Model versioning (keeping track of model evolution)
- Metadata tracking (when trained, by whom, with what data)
- Stage management (development, staging, production)
- Model comparison and lineage tracking

Why model registries matter:

- Prevents losing track of trained models
- Enables rollback to previous versions if needed
- Provides audit trail for compliance
- Facilitates collaboration between team members
- Simplifies deployment process

## Methods

| Method | Summary |
|:-----|:--------|
| `ArchiveModel(String,Int32)` | Archives a model version. |
| `AttachModelCard(String,Int32,ModelCard)` | Attaches a Model Card to a registered model version. |
| `CompareModels(String,Int32,Int32)` | Compares two model versions. |
| `CreateModelVersion(String,IModel<,,>,ModelMetadata<>,String)` | Creates a new version of an existing model. |
| `DeleteModel(String)` | Deletes all versions of a model. |
| `DeleteModelVersion(String,Int32)` | Deletes a specific model version. |
| `GenerateModelCard(String,Int32,String)` | Generates a Model Card from the registered model's metadata and evaluation results. |
| `GetLatestModel(String)` | Gets the latest version of a model. |
| `GetModel(String,Nullable<Int32>)` | Retrieves a specific model version from the registry. |
| `GetModelByStage(String,ModelStage)` | Gets the model currently in a specific stage (e.g., production). |
| `GetModelCard(String,Int32)` | Gets the Model Card for a registered model version. |
| `GetModelLineage(String,Int32)` | Gets the lineage information for a model (how it was created). |
| `GetModelStoragePath(String,Int32)` | Gets the storage location for a model version. |
| `ListModelVersions(String)` | Lists all versions of a specific model. |
| `ListModels(String,Dictionary<String,String>)` | Lists all models in the registry. |
| `RegisterModel(String,IModel<,,>,ModelMetadata<>,Dictionary<String,String>)` | Registers a new model in the registry. |
| `SaveModelCard(String,Int32,String)` | Saves the Model Card for a model version to a file. |
| `SearchModels(ModelSearchCriteria<>)` | Searches for models based on criteria. |
| `TransitionModelStage(String,Int32,ModelStage,Boolean)` | Transitions a model version to a different stage. |
| `UpdateModelMetadata(String,Int32,ModelMetadata<>)` | Updates the metadata for a model version. |
| `UpdateModelTags(String,Int32,Dictionary<String,String>)` | Adds or updates tags for a model. |

