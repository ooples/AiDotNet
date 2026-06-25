---
title: "SuperLearnerOptions"
description: "Configuration options for Super Learner (Stacking) ensemble."
section: "API Reference"
---

`Options & Configuration` · `AiDotNet.Models.Options`

Configuration options for Super Learner (Stacking) ensemble.

## For Beginners

Super Learner is like having a team of experts (base models) vote on
a prediction, but instead of equal votes, a "manager" (meta-learner) learns how to weight
each expert's opinion based on their track record.

**How it works:**

1. Train each base model (e.g., linear regression, random forest, neural network)
2. Get cross-validated predictions from each base model
3. Train a meta-learner to combine these predictions optimally
4. For new data: get predictions from all base models, then combine using meta-learner

**Why it's powerful:**

- Automatically figures out which models work best
- Can combine very different types of models
- Mathematically proven to be optimal (or close to it)
- Less prone to overfitting than simple averaging

**Example:**
You have a linear model (good for simple patterns), a tree model (good for interactions),
and a neural network (good for complex patterns). Super Learner learns to trust the tree
model more for some types of predictions and the neural network for others.

## How It Works

Super Learner is an ensemble method that optimally combines predictions from multiple
base models using a meta-learner. It uses cross-validation to avoid overfitting and
is proven to perform at least as well as the best individual base learner.

Reference: van der Laan, M.J., Polley, E.C., & Hubbard, A.E. (2007). "Super Learner".
Statistical Applications in Genetics and Molecular Biology.

## Properties

| Property | Summary |
|:-----|:--------|
| `IncludeOriginalFeatures` | Gets or sets whether to include the original features in the meta-learner. |
| `MetaLearnerMaxIterations` | Gets or sets the maximum number of iterations for the meta-learner. |
| `MetaLearnerRegularization` | Gets or sets the regularization strength for the meta-learner. |
| `MetaLearnerTolerance` | Gets or sets the tolerance for meta-learner convergence. |
| `MetaLearnerType` | Gets or sets the meta-learner type. |
| `NormalizeBasePredictions` | Gets or sets whether to normalize base model predictions before meta-learning. |
| `NumFolds` | Gets or sets the number of cross-validation folds for generating meta-features. |
| `RetrainOnFullData` | Gets or sets whether to retrain base models on full data after cross-validation. |
| `Seed` | Gets or sets the random seed for reproducibility. |
| `UseStratifiedFolds` | Gets or sets whether to use stratified cross-validation (for classification). |

