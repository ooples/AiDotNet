---
title: "ComponentTypeAttribute"
description: "Specifies the type of an AI pipeline component (Tier 2 metadata)."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Specifies the type of an AI pipeline component (Tier 2 metadata).

## For Beginners

Apply this attribute to components that are part of an AI pipeline
but aren't standalone ML models. Components transform, route, or process data.
You can apply it multiple times if a component serves multiple roles.

## How It Works

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ComponentTypeAttribute(ComponentType)` | Initializes a new instance of the `ComponentTypeAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Type` | Gets the component type. |

