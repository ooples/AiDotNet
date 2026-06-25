---
title: "BarlowTwinsConfig"
description: "Barlow Twins-specific configuration settings."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.SelfSupervisedLearning`

Barlow Twins-specific configuration settings.

## For Beginners

Barlow Twins learns by making the cross-correlation matrix
between embeddings of two views close to the identity matrix (reducing redundancy).

## Properties

| Property | Summary |
|:-----|:--------|
| `Lambda` | Gets or sets the lambda parameter for redundancy reduction. |
| `ProjectionDimension` | Gets or sets the projection dimension. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetConfiguration` | Gets the configuration as a dictionary. |

