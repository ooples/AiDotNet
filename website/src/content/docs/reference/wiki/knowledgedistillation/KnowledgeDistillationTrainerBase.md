---
title: "KnowledgeDistillationTrainerBase<T, TInput, TOutput>"
description: "Abstract base class for all knowledge distillation trainers."
section: "API Reference"
---

`Base Classes` · `AiDotNet.KnowledgeDistillation`

Abstract base class for all knowledge distillation trainers.
Provides common functionality for training loops, data shuffling, validation, and evaluation.

## For Beginners

This base class implements the common training workflow shared by all
distillation trainers. Specific trainer types (standard, self-distillation, online, etc.) inherit
from this and customize only the parts that differ.

## How It Works

**Design Pattern:** Template Method Pattern - the base class defines the training algorithm
structure, and derived classes fill in specific steps.

**Common Functionality Provided:**

- Data shuffling using Fisher-Yates algorithm (O(n) efficiency)
- Epoch and batch management
- Validation after each epoch
- Progress callbacks
- Evaluation metrics (accuracy, loss)
- Teacher and strategy property management

**Derived Classes Override:**

- GetTeacherPredictions(): How to obtain teacher outputs for training
- OnEpochStart(): Custom logic before each epoch
- OnEpochEnd(): Custom logic after each epoch
- OnTrainingStart(): Custom logic before training begins
- OnTrainingEnd(): Custom logic after training completes

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `KnowledgeDistillationTrainerBase(ITeacherModel<,>,IDistillationStrategy<>,DistillationCheckpointConfig,Boolean,Double,Int32,Nullable<Int32>)` | Initializes a new instance of the KnowledgeDistillationTrainerBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `DistillationStrategy` | Gets the distillation strategy for computing loss and gradients. |
| `Teacher` | Gets the teacher model used for distillation. |

## Methods

| Method | Summary |
|:-----|:--------|
| `ArgMax(Vector<>)` | Finds the index of the maximum value in a vector (argmax). |
| `CollectIntermediateActivations(IFullModel<,Tensor<>,Tensor<>>,Tensor<>,HashSet<LayerCategory>)` | Collects intermediate activations from a layered model by performing a forward pass and recording the output of each layer. |
| `ComputeIntermediateActivationLoss(IntermediateActivations<>,IntermediateActivations<>)` | Computes intermediate activation loss between teacher and student for hint-based distillation. |
| `Evaluate(Func<,>,Vector<>,Vector<>)` | Evaluates the student model's accuracy on a dataset. |
| `FisherYatesShuffle(Int32)` | Generates a random permutation of indices using Fisher-Yates shuffle. |
| `GetTeacherPredictions(,Int32)` | Gets teacher predictions for a given input. |
| `IsCorrectPrediction(,)` | Determines if a prediction matches the true label. |
| `OnEpochEnd(Int32,)` | Called after each epoch completes. |
| `OnEpochStart(Int32,Vector<>,Vector<>)` | Called before each epoch starts. |
| `OnTrainingEnd(Vector<>,Vector<>)` | Called after training completes. |
| `OnTrainingStart(Vector<>,Vector<>)` | Called before training starts. |
| `OnValidationComplete(Int32,Double)` | Called after validation completes for an epoch. |
| `ShuffleData(Vector<>,Vector<>)` | Shuffles training data using Fisher-Yates algorithm. |
| `Train(Func<,>,Action<>,Vector<>,Vector<>,Int32,Int32,Action<Int32,>)` | Trains the student model for multiple epochs (interface-compliant overload). |
| `Train(Func<,>,Action<>,Vector<>,Vector<>,Int32,Int32,Vector<>,Vector<>,ICheckpointableModel,Action<Int32,>)` | Trains the student model for multiple epochs using knowledge distillation. |
| `TrainBatch(Func<,>,Action<>,Vector<>,Vector<>)` | Trains the student model on a single batch using knowledge distillation. |

## Fields

| Field | Summary |
|:-----|:--------|
| `NumOps` | Gets the numeric operations helper for the type T. |
| `Random` | Gets the random number generator for data shuffling. |
| `_checkpointConfig` | Checkpoint configuration for automatic model saving during training (internal). |
| `_checkpointManager` | Checkpoint manager for handling checkpoint operations (internal). |
| `_student` | Student model reference for checkpointing (internal). |

