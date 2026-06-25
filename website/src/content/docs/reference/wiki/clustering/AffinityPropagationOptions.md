---
title: "AffinityPropagationOptions<T>"
description: "Configuration options for Affinity Propagation clustering."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Clustering.Options`

Configuration options for Affinity Propagation clustering.

## For Beginners

Affinity Propagation lets data points "vote" for leaders.

Imagine a group trying to choose team captains:

- Everyone can propose themselves as captain
- People vote for who should represent them
- The algorithm finds natural leaders (exemplars)

Key features:

- Doesn't need to know number of clusters beforehand
- Each cluster has a real data point as its center (exemplar)
- Works well when you have a good similarity measure

Main parameter: preference

- Higher preference: More clusters (everyone wants to be a leader)
- Lower preference: Fewer clusters (fewer people volunteer)

## How It Works

Affinity Propagation clusters data by passing messages between pairs of samples
until a high-quality set of exemplars (cluster centers) emerges.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `AffinityPropagationOptions` | Initializes AffinityPropagationOptions with appropriate defaults. |

## Properties

| Property | Summary |
|:-----|:--------|
| `AffinityType` | Gets or sets the affinity type. |
| `ConvergenceIterations` | Gets or sets the number of iterations with no change before stopping. |
| `Copy` | Gets or sets whether to copy the affinity matrix. |
| `Damping` | Gets or sets the damping factor. |
| `DistanceMetric` | Gets or sets the distance metric for Euclidean affinity. |
| `Preference` | Gets or sets the preference value. |

