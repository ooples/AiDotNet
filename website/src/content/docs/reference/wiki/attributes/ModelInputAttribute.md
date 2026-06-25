---
title: "ModelInputAttribute"
description: "Specifies the expected input and output types for a model."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Specifies the expected input and output types for a model.

## For Beginners

Apply this attribute to your model class to declare what type of
data it expects as input and what it produces as output. This makes it easy to discover
which models work with your data format without reading documentation.

## How It Works

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelInputAttribute(Type,Type)` | Initializes a new instance of the `ModelInputAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `InputType` | Gets the expected input type for this model. |
| `OutputType` | Gets the expected output type for this model. |

