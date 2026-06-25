---
title: "YamlConfigurableAttribute"
description: "Marks an interface or abstract base class as discoverable by the YAML configuration system."
section: "API Reference"
---

`Attributes` · `AiDotNet.Configuration`

Marks an interface or abstract base class as discoverable by the YAML configuration system.
The source generator will automatically find all concrete implementations and register them
in the YAML type registry under the specified section name.

## For Beginners

Place this attribute on any interface or abstract class whose
implementations should be configurable via YAML files. The generator will scan the assembly
for all concrete types that implement/extend the marked type and make them available
for YAML-based configuration.

## How It Works

This provides the same YAML discoverability as adding a `Configure*()` method to
`AiModelBuilder`, but without requiring a builder method. Use this when the type
is already used through other APIs and doesn't need its own Configure method.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `YamlConfigurableAttribute(String)` | Initializes a new instance of the `YamlConfigurableAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `SectionName` | Gets the YAML section name for this type family. |

