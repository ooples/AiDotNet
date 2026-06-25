---
title: "LeafFederatedDatasetLoadOptions"
description: "Options controlling how LEAF federated benchmark JSON files are loaded."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.FederatedLearning.Benchmarks.Leaf`

Options controlling how LEAF federated benchmark JSON files are loaded.

## For Beginners

Think of these as "load settings" for a dataset file:
you can choose to load all clients/users or just the first N to keep the run fast.

## How It Works

LEAF datasets can be large. These options allow callers to load a smaller subset for
quick experiments and CI-friendly tests.

## Properties

| Property | Summary |
|:-----|:--------|
| `MaxUsers` | Gets or sets the maximum number of users/clients to load (null loads all users). |
| `ValidateDeclaredSampleCounts` | Gets or sets whether to validate that each user's declared `num_samples` matches the actual sample count. |

