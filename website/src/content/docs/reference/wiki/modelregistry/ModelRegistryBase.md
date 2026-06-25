---
title: "ModelRegistryBase<T, TInput, TOutput>"
description: "Base class for model registry implementations."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ModelRegistry`

Base class for model registry implementations.

## How It Works

**For Beginners:** This abstract base class provides common functionality for model
registry systems. It handles storage path management, versioning logic, and stage transitions
while leaving specific storage implementation to derived classes.

Key features:

- Path security validation
- Model versioning support
- Stage transition management (Development, Staging, Production, Archived)
- Thread-safe model tracking

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelRegistryBase(String,String)` | Initializes a new instance of the ModelRegistryBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArchiveModel(String,Int32)` | Archives a model version. |
| `AttachModelCard(String,Int32,ModelCard)` | Attaches a Model Card to a registered model version. |
| `CompareModels(String,Int32,Int32)` | Compares two model versions. |
| `CreateModelVersion(String,IModel<,,>,ModelMetadata<>,String)` | Creates a new version of an existing model. |
| `DeleteModel(String)` | Deletes all versions of a model. |
| `DeleteModelVersion(String,Int32)` | Deletes a specific model version. |
| `DeserializeFromJson(String)` | Deserializes a JSON string to an object. |
| `EnsureRegistryDirectoryExists` | Ensures the registry directory exists. |
| `GenerateModelCard(String,Int32,String)` | Generates a Model Card from the registered model's metadata and evaluation results. |
| `GetLatestModel(String)` | Gets the latest version of a model. |
| `GetModel(String,Nullable<Int32>)` | Retrieves a specific model version from the registry. |
| `GetModelByStage(String,ModelStage)` | Gets the model currently in a specific stage. |
| `GetModelCard(String,Int32)` | Gets the Model Card for a registered model version. |
| `GetModelDirectoryPath(String)` | Gets the directory path for a model. |
| `GetModelLineage(String,Int32)` | Gets the lineage information for a model. |
| `GetModelStoragePath(String,Int32)` | Gets the storage location for a model version. |
| `GetModelVersionPath(String,Int32)` | Gets the file path for a specific model version. |
| `GetSanitizedFileName(String)` | Sanitizes a file name to prevent path traversal attacks. |
| `GetSanitizedPath(String,String)` | Gets a sanitized path, ensuring it doesn't escape the base directory. |
| `ListModelVersions(String)` | Lists all versions of a specific model. |
| `ListModels(String,Dictionary<String,String>)` | Lists all models in the registry. |
| `RegisterModel(String,IModel<,,>,ModelMetadata<>,Dictionary<String,String>)` | Registers a new model in the registry. |
| `SaveModelCard(String,Int32,String)` | Saves the Model Card for a model version to a file. |
| `SearchModels(ModelSearchCriteria<>)` | Searches for models based on criteria. |
| `SerializeToJson(Object)` | Serializes an object to JSON. |
| `TransitionModelStage(String,Int32,ModelStage,Boolean)` | Transitions a model version to a different stage. |
| `UpdateModelMetadata(String,Int32,ModelMetadata<>)` | Updates the metadata for a model version. |
| `UpdateModelTags(String,Int32,Dictionary<String,String>)` | Adds or updates tags for a model. |
| `ValidateModelName(String)` | Validates that a model name is valid. |
| `ValidatePathWithinDirectory(String,String)` | Validates that a path is within the specified directory. |

## Fields

| Field | Summary |
|:-----|:--------|
| `JsonSettings` | JSON serialization settings for consistent serialization. |
| `RegistryDirectory` | The directory where models are stored. |
| `SyncLock` | Lock object for thread-safe operations. |

