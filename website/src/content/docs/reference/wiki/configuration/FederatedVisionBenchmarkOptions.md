---
title: "FederatedVisionBenchmarkOptions"
description: "Configuration options for federated vision benchmark suites."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Configuration`

Configuration options for federated vision benchmark suites.

## For Beginners

Vision benchmarks test models on image-like data. You select a suite (enum)
and provide the minimal dataset configuration here.

## How It Works

This groups dataset-specific options for vision benchmarks (for example, FEMNIST and CIFAR) under a single
facade-facing configuration object.

## Properties

| Property | Summary |
|:-----|:--------|
| `Cifar10` | Gets or sets CIFAR-10 options (CIFAR binary files with synthetic federated partitioning). |
| `Cifar100` | Gets or sets CIFAR-100 options (CIFAR-100 binary files with synthetic federated partitioning). |
| `Femnist` | Gets or sets FEMNIST options (LEAF JSON split files). |

