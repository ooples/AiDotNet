---
title: "ExperimentTrackerBase<T>"
description: "Base class for experiment tracking implementations that manage ML experiments and runs."
section: "API Reference"
---

`Base Classes` · `AiDotNet.ExperimentTracking`

Base class for experiment tracking implementations that manage ML experiments and runs.

## How It Works

**For Beginners:** This abstract base class provides common functionality for experiment
tracking systems. It handles storage path management, security validation, and JSON serialization
while leaving the specific storage implementation to derived classes.

Benefits of this architecture:

- Consistent path security across all experiment tracker implementations
- Shared JSON serialization settings
- Common helper methods for file name sanitization
- Extensible design for different storage backends

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ExperimentTrackerBase(String,String)` | Initializes a new instance of the ExperimentTrackerBase class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateExperiment(String,String,Dictionary<String,String>)` | Creates a new experiment to organize related training runs. |
| `DeleteExperiment(String)` | Deletes an experiment and all its associated runs. |
| `DeleteRun(String)` | Deletes a specific run. |
| `DeserializeFromJson(String)` | Deserializes a JSON string to an object. |
| `EnsureStorageDirectoryExists` | Ensures the storage directory exists. |
| `GetExperiment(String)` | Gets an existing experiment by its ID. |
| `GetExperimentDirectoryPath(String)` | Gets the directory path for an experiment. |
| `GetRun(String)` | Gets an existing run by its ID. |
| `GetSanitizedFileName(String)` | Sanitizes a file name to prevent path traversal attacks. |
| `GetSanitizedPath(String,String)` | Gets a sanitized path, ensuring it doesn't escape the base directory. |
| `ListExperiments(String)` | Lists all experiments, optionally filtered by criteria. |
| `ListRuns(String,String)` | Lists all runs in an experiment, optionally filtered by criteria. |
| `SearchRuns(String,Int32)` | Searches for runs across all experiments based on criteria. |
| `SerializeToJson(Object)` | Serializes an object to JSON. |
| `StartRun(String,String,Dictionary<String,String>)` | Starts a new training run within an experiment. |
| `ValidatePathWithinDirectory(String,String)` | Validates that a path is within the specified directory. |

## Fields

| Field | Summary |
|:-----|:--------|
| `JsonSettings` | JSON serialization settings for consistent serialization across all trackers. |
| `StorageDirectory` | The directory where experiment data is stored. |
| `SyncLock` | Lock object for thread-safe operations. |

