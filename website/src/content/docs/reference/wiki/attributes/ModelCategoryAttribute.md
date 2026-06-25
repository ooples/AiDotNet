---
title: "ModelCategoryAttribute"
description: "Specifies the algorithm family or category that a model belongs to."
section: "API Reference"
---

`Attributes` · `AiDotNet.Attributes`

Specifies the algorithm family or category that a model belongs to.

## For Beginners

Apply this attribute to your model class to indicate what kind
of algorithm it uses. You can apply it multiple times if the model combines
multiple algorithm families (e.g., a Transformer that is also an Autoencoder).

## How It Works

**Usage:**

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `ModelCategoryAttribute(ModelCategory)` | Initializes a new instance of the `ModelCategoryAttribute` class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `Category` | Gets the algorithm category this model belongs to. |

