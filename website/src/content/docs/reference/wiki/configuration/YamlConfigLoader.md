---
title: "YamlConfigLoader"
description: "Loads and deserializes YAML configuration files into strongly-typed configuration objects."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Configuration`

Loads and deserializes YAML configuration files into strongly-typed configuration objects.

## For Beginners

This class reads a YAML file from disk (or a YAML string)
and converts it into a structured C# object that the builder or trainer can use. YAML uses
camelCase property names (e.g., `timeSeriesModel`, `gpuAcceleration`).

## How It Works

**Example usage:**

## Methods

| Method | Summary |
|:-----|:--------|
| `LoadFromFile(String)` | Loads a YAML configuration file from disk and deserializes it into a `YamlModelConfig`. |
| `LoadFromFile(String)` | Loads a YAML configuration file from disk and deserializes it into the specified type. |
| `LoadFromString(String)` | Deserializes a YAML string into a `YamlModelConfig`. |
| `LoadFromString(String)` | Deserializes a YAML string into the specified type. |

