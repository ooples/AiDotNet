---
title: "FitnessCalculatorBase<T, TInput, TOutput>"
description: "Base class for all fitness calculators that evaluate how well a model performs."
section: "API Reference"
---

`Base Classes` · `AiDotNet.FitnessCalculators`

Base class for all fitness calculators that evaluate how well a model performs.

## For Beginners

This is a foundation class that all fitness calculators build upon.

Think of a fitness calculator like a judge in a competition:

- It looks at how your AI model performed (its predictions vs. the actual answers)
- It gives a score based on specific criteria (like accuracy, error rate, etc.)
- It helps you determine if one model is better than another

Different fitness calculators judge models in different ways, just like different 
sports have different scoring systems. Some calculators consider higher scores better 
(like accuracy), while others consider lower scores better (like error rates).

This base class provides the common functionality that all these different "judges" share.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `FitnessCalculatorBase(Boolean,DataSetType)` | Initializes a new instance of the FitnessCalculatorBase class. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsHigherScoreBetter` | Gets a value indicating whether higher fitness scores represent better performance. |
| `PreferredDataSetType` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `CalculateFitnessScore(DataSetStats<,,>)` | Calculates the fitness score using a specific dataset. |
| `CalculateFitnessScore(ModelEvaluationData<,,>)` | Calculates the fitness score for a model using the specified evaluation data. |
| `GetFitnessScore(DataSetStats<,,>)` | Abstract method that must be implemented by derived classes to calculate the specific fitness score. |
| `IsBetterFitness(,)` | Determines whether a new fitness score is better than the current best score. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_dataSetType` | Specifies which dataset (training, validation, or testing) to use for fitness calculation. |
| `_isHigherScoreBetter` | Indicates whether higher fitness scores represent better performance. |
| `_numOps` | Provides mathematical operations for the specific numeric type being used. |

