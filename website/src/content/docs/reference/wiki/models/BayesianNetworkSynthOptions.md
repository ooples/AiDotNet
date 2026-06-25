---
title: "BayesianNetworkSynthOptions<T>"
description: "Configuration options for Bayesian Network Synthesis, a statistical approach that learns a directed acyclic graph (DAG) structure and conditional probability tables to generate synthetic tabular data via ancestral sampling."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Bayesian Network Synthesis, a statistical approach that
learns a directed acyclic graph (DAG) structure and conditional probability tables
to generate synthetic tabular data via ancestral sampling.

## For Beginners

This method creates a probabilistic model of your data:

Think of a family tree of features — some features "depend on" others.
For example, in a health dataset:

1. Age has no parents (sampled first)
2. Blood pressure depends on Age
3. Medication depends on Blood pressure

The model learns these dependency chains and samples new data following
the same parent-to-child order, producing statistically coherent rows.

Unlike neural network generators (CTGAN, TVAE), this uses classical statistics,
making it faster to train and more interpretable, though less flexible for
complex distributions.

Example:

## How It Works

Bayesian Network Synthesis operates in three phases:

- **Structure learning**: Discovers a DAG using greedy hill-climbing with BIC scoring
- **Parameter estimation**: Estimates conditional probability tables (CPTs) from the data
- **Ancestral sampling**: Generates data by sampling from root nodes down through the DAG

## Properties

| Property | Summary |
|:-----|:--------|
| `LaplaceSmoothing` | Gets or sets the Laplace smoothing constant for CPT estimation. |
| `MaxIterations` | Gets or sets the maximum number of structure learning iterations. |
| `MaxParents` | Gets or sets the maximum number of parents per node in the DAG. |
| `NumBins` | Gets or sets the number of discretization bins for continuous features. |

