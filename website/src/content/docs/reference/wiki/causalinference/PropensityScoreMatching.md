---
title: "PropensityScoreMatching<T>"
description: "Implements Propensity Score Matching for causal effect estimation."
section: "API Reference"
---

`Models & Types` · `AiDotNet.CausalInference`

Implements Propensity Score Matching for causal effect estimation.

## For Beginners

PSM works by finding "twins" - pairs of people who were equally
likely to be treated but one actually was and one wasn't. By comparing these matched
pairs, we can estimate the causal effect of treatment.

How it works:

1. Estimate propensity scores (probability of treatment for each person)
2. For each treated person, find a control person with the most similar propensity score
3. Compare outcomes between matched pairs
4. Average the differences to get the treatment effect

The key insight: If two people have the same propensity score, the "assignment"
of treatment is essentially random between them. This mimics a randomized experiment.

Example - Job Training Study:

- Person A: propensity=0.7, treated=yes, salary=$50,000
- Person B: propensity=0.7, treated=no, salary=$45,000
- Estimated effect: $50,000 - $45,000 = $5,000

Advantages:

- Intuitive and easy to explain
- Creates a matched sample that looks like a mini experiment
- Can visually verify match quality

Limitations:

- Discards unmatched observations
- Sensitive to match quality
- Only adjusts for measured confounders

References:

- Rosenbaum & Rubin (1983). "The Central Role of the Propensity Score"

## How It Works

Propensity Score Matching (PSM) estimates treatment effects by matching treated individuals
to control individuals with similar propensity scores (probability of treatment).

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `PropensityScoreMatching(Double,Boolean,Int32,Nullable<Int32>)` | Gets the model type. |

## Methods

| Method | Summary |
|:-----|:--------|
| `CreateNewInstance` | Creates a new instance of this type. |
| `EstimateATE(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect using matched pairs. |
| `EstimateATT(Matrix<>,Vector<Int32>,Vector<>)` | Estimates the Average Treatment Effect on the Treated. |
| `EstimateCATEPerIndividual(Matrix<>,Vector<Int32>,Vector<>)` | Estimates individual treatment effects using matched pairs. |
| `EstimatePropensityScoresCore(Matrix<>)` | Estimates propensity scores using the fitted model. |
| `EstimateTreatmentEffect(Matrix<>)` | Estimates treatment effects for individuals using matched pairs. |
| `Fit(Matrix<>,Vector<>,Vector<>)` | Fits the causal model using the ICausalModel interface signature. |
| `Fit(Matrix<>,Vector<Int32>)` | Fits the propensity score model to the data. |
| `GetAdditionalModelData` | Gets additional model data for serialization. |
| `GetMatchQuality(Matrix<>,Vector<Int32>)` | Gets match quality statistics (balance check). |
| `GetNumberOfMatches(Matrix<>,Vector<Int32>)` | Gets the number of matched pairs. |
| `GetParameters` | Gets the model parameters (propensity score coefficients). |
| `LoadAdditionalModelData(JObject)` | Loads additional model data from deserialization. |
| `PerformMatching(Vector<>,Vector<Int32>)` | Performs the actual matching of treated to control individuals. |
| `Predict(Matrix<>)` | Standard prediction - returns propensity scores. |
| `PredictControl(Matrix<>)` | Predicts outcomes under control for the given features. |
| `PredictTreated(Matrix<>)` | Predicts outcomes under treatment for the given features. |
| `PredictTreatmentEffect(Matrix<>)` | Predicts treatment effects for new individuals. |
| `SetParameters(Vector<>)` | Sets the model parameters. |
| `WithParameters(Vector<>)` | Creates a new instance with specified parameters. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_cachedFeatures` | Cached feature matrix from fitting. |
| `_cachedOutcome` | Cached outcome vector from fitting. |
| `_cachedTreatment` | Cached treatment vector from fitting. |
| `_caliper` | The caliper width for matching (maximum allowed propensity score difference). |
| `_matchRatio` | Number of matches per treated individual. |
| `_propensityCoefficients` | Stores the logistic regression coefficients for propensity score estimation. |
| `_random` | Random number generator for tie-breaking. |
| `_withReplacement` | Whether to use replacement in matching. |

