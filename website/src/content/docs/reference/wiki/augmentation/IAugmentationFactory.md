---
title: "IAugmentationFactory<T, TData>"
description: "Factory for creating augmentations from configuration."
section: "API Reference"
---

`Interfaces` · `AiDotNet.Augmentation`

Factory for creating augmentations from configuration.

## How It Works

Enables dynamic augmentation creation from serialized configurations.

## Properties

| Property | Summary |
|:-----|:--------|
| `RegisteredTypes` | Gets all registered augmentation types. |

## Methods

| Method | Summary |
|:-----|:--------|
| `Create(String,IDictionary<String,Object>)` | Creates an augmentation from type name and parameters. |
| `GetSearchSpace(String)` | Gets the search space for an augmentation type. |
| `Register(String,Func<IDictionary<String,Object>,IAugmentation<,>>)` | Registers a custom augmentation type. |

