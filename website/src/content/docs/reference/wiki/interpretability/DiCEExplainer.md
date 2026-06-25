---
title: "DiCEExplainer<T>"
description: "DiCE (Diverse Counterfactual Explanations) explainer using genetic algorithm-based search."
section: "API Reference"
---

`Models & Types` · `AiDotNet.Interpretability.Explainers`

DiCE (Diverse Counterfactual Explanations) explainer using genetic algorithm-based search.

## For Beginners

DiCE generates MULTIPLE diverse counterfactual explanations,
not just one. This is much more useful in practice because:

1. **Multiple options:** "You could get a loan by EITHER increasing income OR

reducing debt" instead of just one path

2. **Diversity:** The counterfactuals are intentionally different from each other
3. **Actionability:** Respects constraints like "you can't become younger"
4. **Realism:** Changes are minimal and realistic

**How DiCE works:**
Uses a genetic algorithm to evolve a population of candidate counterfactuals,
optimizing for:

- **Validity:** Must achieve the target prediction
- **Proximity:** Stay close to the original instance
- **Sparsity:** Change as few features as possible
- **Diversity:** Counterfactuals should differ from each other

**Example output:**
Original: Loan denied (income=$40k, debt=$30k, employed=No)
Counterfactual 1: Loan approved (income=$50k, debt=$30k, employed=No)
Counterfactual 2: Loan approved (income=$40k, debt=$20k, employed=No)
Counterfactual 3: Loan approved (income=$40k, debt=$30k, employed=Yes)

Each shows a DIFFERENT way to get approved!

## Constructors

| Constructor | Summary |
|:-----|:--------|
| `DiCEExplainer(Func<Matrix<>,Vector<>>,Int32,Int32,Int32,Int32,Double,Double,Double,Double,Double,Double,String[],[],[],Boolean[],FeatureType[],Nullable<Int32>)` | Initializes a new DiCE explainer. |

## Properties

| Property | Summary |
|:-----|:--------|
| `IsGPUAccelerated` |  |
| `MethodName` |  |
| `SupportsGlobalExplanations` |  |
| `SupportsLocalExplanations` |  |

## Methods

| Method | Summary |
|:-----|:--------|
| `ApplyCrossover(List<Vector<>>,Vector<>,Random)` | Applies crossover between selected individuals. |
| `ApplyMutation(List<Vector<>>,Vector<>,Random)` | Applies mutation to offspring. |
| `CombineWithElitism(List<Vector<>>,List<Vector<>>,Double[],Vector<>,Double)` | Combines parents and offspring with elitism. |
| `ComputeDiversityScore(Vector<>,List<Vector<>>,Vector<>)` | Computes diversity score for an individual. |
| `ComputeFitness(Vector<>,Vector<>,Double,List<Vector<>>)` | Computes fitness for a single individual. |
| `ComputeNormalizedDistance(Vector<>,Vector<>)` | Computes normalized distance from original. |
| `ComputeSetDiversity(List<Vector<>>,Vector<>)` | Computes diversity of a set of counterfactuals. |
| `ComputeSparsity(Vector<>,Vector<>)` | Computes sparsity (fraction of features changed). |
| `ComputeValidityLoss(Double,Double)` | Computes validity loss (how far from target prediction). |
| `CountValidCounterfactuals(List<Vector<>>,Double)` | Counts valid counterfactuals that reach the target. |
| `CreateSingleCounterfactual(Vector<>,Vector<>,)` | Creates a single counterfactual result. |
| `EvaluatePopulation(List<Vector<>>,Vector<>,Double)` | Evaluates fitness of all individuals in the population. |
| `Explain(Vector<>)` | Generates diverse counterfactual explanations for an instance. |
| `ExplainBatch(Matrix<>)` |  |
| `ExplainWithTarget(Vector<>,Double)` | Generates diverse counterfactual explanations for a specific target. |
| `InitializePopulation(Vector<>,Random)` | Initializes the population with random perturbations. |
| `SelectDiverseCounterfactuals(List<Vector<>>,Double[],Vector<>,Double)` | Selects diverse counterfactuals using greedy selection. |
| `SelectTopCounterfactuals(List<Vector<>>,Double[],Vector<>,Double)` | Selects top counterfactuals based on fitness. |
| `SetGPUHelper(GPUExplainerHelper<>)` |  |
| `SinglePointCrossover(Vector<>,Vector<>,Vector<>,Random)` | Single-point crossover between two parents. |
| `TournamentSelection(List<Vector<>>,Double[],Random)` | Tournament selection for genetic algorithm. |

