---
title: "YamlJsonSchema"
description: "Generates a JSON Schema for the AiDotNet YAML configuration system."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Configuration`

Generates a JSON Schema for the AiDotNet YAML configuration system.
The schema provides IntelliSense/autocomplete when editing YAML files in VS Code
(with the YAML Language Server extension) and validates configuration structure.

## For Beginners

JSON Schema tells your editor what properties are valid
in a YAML file, what values they accept, and provides descriptions on hover.
To use it, add this comment at the top of your YAML file:

## How It Works

Then generate the schema file by calling:

## Methods

| Method | Summary |
|:-----|:--------|
| `Generate` | Generates a complete JSON Schema for the AiDotNet YAML configuration. |

