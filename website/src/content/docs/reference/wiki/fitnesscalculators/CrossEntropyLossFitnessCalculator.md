---
title: "CrossEntropyLossFitnessCalculator<T, TInput, TOutput>"
description: "A fitness calculator that uses Cross Entropy Loss to evaluate model performance for classification tasks."
section: "API Reference"
---

`Models & Types` · `AiDotNet.FitnessCalculators`

A fitness calculator that uses Cross Entropy Loss to evaluate model performance for classification tasks.

## For Beginners

This calculator helps you evaluate how well your model is performing on classification tasks,
where you need to assign items to specific categories or classes.

Cross Entropy Loss is one of the most common ways to measure how well a model is doing when it needs to
choose between multiple options (like identifying if an image contains a cat, dog, or bird).

Think of it like a test where:

- Your model gives a confidence score for each possible answer (e.g., "I'm 80% sure this is a cat")
- The correct answer is known (e.g., "This is actually a cat")
- The loss measures how far off your model's confidence was from being perfectly correct

Some common applications include:

- Image classification (identifying objects in images)
- Sentiment analysis (determining if text is positive, negative, or neutral)
- Medical diagnosis (classifying medical conditions)
- Spam detection (determining if an email is spam or not)

Lower values mean your model is more confident about the correct answers and less confident about wrong answers.
A perfect model would have a loss of 0, while a completely wrong model would have a very high loss.

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `CrossEntropyLossFitnessCalculator(DataSetType)` | Initializes a new instance of the CrossEntropyLossFitnessCalculator class. |

## Methods

| Method | Summary |
|:-----|:--------|
| `GetFitnessScore(DataSetStats<,,>)` | Calculates the Cross Entropy Loss between predicted and actual values. |

