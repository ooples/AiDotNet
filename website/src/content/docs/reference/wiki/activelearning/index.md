---
title: "Active Learning"
description: "All 93 public types in the AiDotNet.activelearning namespace, organized by kind."
section: "API Reference"
---

**93** public types in this namespace, organized by kind.

## Models & Types (47)

| Type | Summary |
|:-----|:--------|
| [`ActiveLearner<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/activelearner/) | Core implementation of the active learner that orchestrates the active learning loop. |
| [`ActiveLearningContext<T>`](/docs/reference/wiki/activelearning/activelearningcontext/) | Context information for stopping criterion evaluation. |
| [`ActiveLearningIterationResult<T>`](/docs/reference/wiki/activelearning/activelearningiterationresult/) | Result from a single active learning iteration. |
| [`ActiveLearningResult<T>`](/docs/reference/wiki/activelearning/activelearningresult/) | Final result from the complete active learning process. |
| [`BADGEStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/badgestrategy/) | BADGE (Batch Active learning by Diverse Gradient Embeddings) strategy. |
| [`BALDStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/baldstrategy/) | Bayesian Active Learning by Disagreement (BALD) strategy. |
| [`BALD<T>`](/docs/reference/wiki/activelearning/bald/) | Implements Bayesian Active Learning by Disagreement (BALD) for sample selection. |
| [`BatchBALD<T>`](/docs/reference/wiki/activelearning/batchbald/) | Implements BatchBALD for joint batch selection in active learning. |
| [`BudgetExhaustedCriterion<T>`](/docs/reference/wiki/activelearning/budgetexhaustedcriterion/) | Stopping criterion based on labeling budget exhaustion. |
| [`ClusteredBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/clusteredbatchstrategy/) | Clustering-based batch selection strategy using k-means clustering. |
| [`ColdStartAnalysisResult<T>`](/docs/reference/wiki/activelearning/coldstartanalysisresult/) | Cold start analysis result showing initial sample selection performance. |
| [`CompositeCriterion<T>`](/docs/reference/wiki/activelearning/compositecriterion/) | Composite stopping criterion that combines multiple criteria. |
| [`ConvergenceCriterion<T>`](/docs/reference/wiki/activelearning/convergencecriterion/) | Stopping criterion based on learning curve convergence. |
| [`CoreSetSelection<T>`](/docs/reference/wiki/activelearning/coresetselection/) | Implements core-set selection using the k-center-greedy algorithm for active learning. |
| [`CoreSetStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/coresetstrategy/) | CoreSet strategy for active learning using geometric diversity. |
| [`DensityWeightedSampling<T>`](/docs/reference/wiki/activelearning/densityweightedsampling/) | Implements density-weighted sampling for active learning. |
| [`DiversitySampling<T>`](/docs/reference/wiki/activelearning/diversitysampling/) | Implements Diversity Sampling for active learning. |
| [`EntropySampling<T>`](/docs/reference/wiki/activelearning/entropysampling/) | Implements entropy sampling for active learning sample selection. |
| [`EvaluationMetrics<T>`](/docs/reference/wiki/activelearning/evaluationmetrics/) | Evaluation metrics from model evaluation. |
| [`EvaluationResult<T>`](/docs/reference/wiki/activelearning/evaluationresult/) | Result of evaluating a model. |
| [`ExpectedModelChange<T>`](/docs/reference/wiki/activelearning/expectedmodelchange/) | Implements Expected Model Change (EMC) for active learning. |
| [`GradientBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/gradientbatchstrategy/) | Gradient-based batch selection strategy using gradient embeddings (BADGE-style). |
| [`HybridSampling<T>`](/docs/reference/wiki/activelearning/hybridsampling/) | Implements Hybrid Sampling that combines multiple active learning strategies. |
| [`InMemoryDataset<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/inmemorydataset/) | In-memory implementation of a dataset for active learning. |
| [`InformationDensity<T>`](/docs/reference/wiki/activelearning/informationdensity/) | Implements information density sampling for active learning. |
| [`LearningCurve<T>`](/docs/reference/wiki/activelearning/learningcurve/) | Represents a learning curve showing performance vs. |
| [`LeastConfidenceSampling<T>`](/docs/reference/wiki/activelearning/leastconfidencesampling/) | Implements least confidence sampling for active learning sample selection. |
| [`MarginSampling<T>`](/docs/reference/wiki/activelearning/marginsampling/) | Implements margin sampling for active learning sample selection. |
| [`PerformancePlateauCriterion<T>`](/docs/reference/wiki/activelearning/performanceplateaucriterion/) | Stopping criterion based on performance plateau detection. |
| [`PredictionStabilityCriterion<T>`](/docs/reference/wiki/activelearning/predictionstabilitycriterion/) | Stopping criterion based on prediction stability across iterations. |
| [`QueryByCommitteeStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/querybycommitteestrategy/) | Query By Committee (QBC) strategy for active learning. |
| [`QueryByCommittee<T>`](/docs/reference/wiki/activelearning/querybycommittee/) | Implements Query-by-Committee (QBC) for active learning. |
| [`QueryQualityCriterion<T>`](/docs/reference/wiki/activelearning/queryqualitycriterion/) | Stopping criterion based on quality of remaining query candidates. |
| [`QueryStrategyMetrics<T>`](/docs/reference/wiki/activelearning/querystrategymetrics/) | Detailed metrics for query strategy performance analysis. |
| [`RandomSampling<T>`](/docs/reference/wiki/activelearning/randomsampling/) | Implements random sampling for active learning (baseline strategy). |
| [`RankedBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/rankedbatchstrategy/) | Simple ranked batch selection strategy with diversity filtering. |
| [`SamplesSelectedEventArgs<TInput>`](/docs/reference/wiki/activelearning/samplesselectedeventargs/) | Event arguments for when samples are selected for labeling. |
| [`StrategyComparisonResult<T>`](/docs/reference/wiki/activelearning/strategycomparisonresult/) | Comparison result between different query strategies. |
| [`SubmodularBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/submodularbatchstrategy/) | Submodular batch selection strategy using facility location objectives. |
| [`TimeBudgetCriterion<T>`](/docs/reference/wiki/activelearning/timebudgetcriterion/) | Stopping criterion based on time budget exhaustion. |
| [`TrainingMetrics<T>`](/docs/reference/wiki/activelearning/trainingmetrics/) | Training metrics from a model training iteration. |
| [`TrainingResult<T>`](/docs/reference/wiki/activelearning/trainingresult/) | Result of training a model. |
| [`UncertaintySamplingStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/uncertaintysamplingstrategy/) | Uncertainty sampling strategy for active learning. |
| [`UncertaintySampling<T>`](/docs/reference/wiki/activelearning/uncertaintysampling/) | Implements uncertainty sampling for active learning. |
| [`UncertaintyThresholdCriterion<T>`](/docs/reference/wiki/activelearning/uncertaintythresholdcriterion/) | Stopping criterion based on model uncertainty reaching a threshold. |
| [`UnlabeledPoolExhaustedCriterion<T>`](/docs/reference/wiki/activelearning/unlabeledpoolexhaustedcriterion/) | Stopping criterion that triggers when the unlabeled pool is exhausted. |
| [`VariationRatios<T>`](/docs/reference/wiki/activelearning/variationratios/) | Implements variation ratios for active learning sample selection. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`DiversityStrategyBase<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/diversitystrategybase/) | Base class for diversity strategies with common functionality. |

## Interfaces (22)

| Type | Summary |
|:-----|:--------|
| [`IActiveLearner<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/iactivelearner/) | Interface for active learners. |
| [`IBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/ibatchstrategy/) | Interface for batch selection strategies in active learning. |
| [`IBayesianStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/ibayesianstrategy/) | Interface for Bayesian query strategies (e.g., BALD). |
| [`IClusteringBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/iclusteringbatchstrategy/) | Interface for clustering-based batch selection. |
| [`ICommitteeStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/icommitteestrategy/) | Interface for committee-based query strategies (Query By Committee). |
| [`ICompositeCriterion<T>`](/docs/reference/wiki/activelearning/icompositecriterion/) | Interface for composite stopping criteria (multiple criteria combined). |
| [`IDensityWeightedStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/idensityweightedstrategy/) | Interface for density-weighted query strategies. |
| [`IDiversityStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/idiversitystrategy/) | Interface for diversity-based sampling strategies in active learning. |
| [`IDropoutModel<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/idropoutmodel/) | Interface for models that support dropout inference. |
| [`IFeatureExtractor<T, TInput>`](/docs/reference/wiki/activelearning/ifeatureextractor/) | Interface for models that can extract feature representations from inputs. |
| [`IFeatureExtractor<T, TInput>`](/docs/reference/wiki/activelearning/ifeatureextractor-2/) | Interface for models that can extract feature representations. |
| [`IGradientBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/igradientbatchstrategy/) | Interface for gradient-based batch selection (e.g., BADGE). |
| [`IGradientModel<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/igradientmodel/) | Interface for models that can compute gradient embeddings. |
| [`IOracle<TInput, TOutput>`](/docs/reference/wiki/activelearning/ioracle/) | Interface for oracles (labeling providers) in active learning. |
| [`IPredictionBasedCriterion<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/ipredictionbasedcriterion/) | Interface for stopping criteria that need prediction access. |
| [`IProbabilisticModel<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/iprobabilisticmodel/) | Interface for models that can provide probability distributions. |
| [`IQueryStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/iquerystrategy/) | Interface for query strategies in active learning. |
| [`IStoppingCriterion<T>`](/docs/reference/wiki/activelearning/istoppingcriterion/) | Interface for stopping criteria in active learning. |
| [`ISubmodularBatchStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/isubmodularbatchstrategy/) | Interface for submodular batch selection strategies. |
| [`ITrainableModel<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/itrainablemodel/) | Interface for models that support training on datasets. |
| [`IUncertaintyBasedCriterion<T>`](/docs/reference/wiki/activelearning/iuncertaintybasedcriterion/) | Interface for uncertainty-based stopping criteria. |
| [`IUncertaintyStrategy<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/iuncertaintystrategy/) | Interface for query strategies that support uncertainty-based selection. |

## Enums (21)

| Type | Summary |
|:-----|:--------|
| [`ChangeMetric<T>`](/docs/reference/wiki/activelearning/changemetric/) | Defines the change metric to use. |
| [`ClusteringMethod`](/docs/reference/wiki/activelearning/clusteringmethod/) | Clustering methods for diversity-based sampling. |
| [`ColdStartStrategy`](/docs/reference/wiki/activelearning/coldstartstrategy/) | Strategies for cold-start sample selection. |
| [`CombinationMethod<T>`](/docs/reference/wiki/activelearning/combinationmethod/) | Defines methods for combining strategy scores. |
| [`CombinationMode`](/docs/reference/wiki/activelearning/combinationmode/) | Mode for combining multiple stopping criteria. |
| [`CommitteeDisagreementMeasure`](/docs/reference/wiki/activelearning/committeedisagreementmeasure/) | Disagreement measures for Query By Committee. |
| [`DisagreementMeasure`](/docs/reference/wiki/activelearning/disagreementmeasure/) | Methods for measuring disagreement in a committee of models. |
| [`DisagreementMeasure<T>`](/docs/reference/wiki/activelearning/disagreementmeasure-2/) | Defines the disagreement measure to use. |
| [`DistanceMetric`](/docs/reference/wiki/activelearning/distancemetric/) | Distance metrics for diversity-based sampling. |
| [`DistanceMetric<T>`](/docs/reference/wiki/activelearning/distancemetric-2/) | Defines the distance metric to use for core-set selection. |
| [`DistanceMetric<T>`](/docs/reference/wiki/activelearning/distancemetric-3/) | Defines the distance metric to use. |
| [`DistanceMetric`](/docs/reference/wiki/activelearning/distancemetric-4/) | Distance metrics for diversity-based strategies. |
| [`DiversityMethod<T>`](/docs/reference/wiki/activelearning/diversitymethod/) | Defines the diversity selection method. |
| [`GradientApproximation`](/docs/reference/wiki/activelearning/gradientapproximation/) | Methods for approximating gradient-based importance. |
| [`QueryStrategyType`](/docs/reference/wiki/activelearning/querystrategytype/) | Types of query strategies for active learning. |
| [`SimilarityMeasure<T>`](/docs/reference/wiki/activelearning/similaritymeasure/) | Defines the similarity measure for computing information density. |
| [`StabilityMeasure`](/docs/reference/wiki/activelearning/stabilitymeasure/) | Stability measurement methods for prediction stability criterion. |
| [`StoppingCriterionType`](/docs/reference/wiki/activelearning/stoppingcriteriontype/) | Criteria for early stopping in active learning. |
| [`SubmodularObjective`](/docs/reference/wiki/activelearning/submodularobjective/) | Types of submodular objective functions. |
| [`UncertaintyMeasure`](/docs/reference/wiki/activelearning/uncertaintymeasure/) | Methods for measuring uncertainty in predictions. |
| [`UncertaintyMeasure<T>`](/docs/reference/wiki/activelearning/uncertaintymeasure-2/) | Defines the uncertainty measure to use. |

## Options & Configuration (1)

| Type | Summary |
|:-----|:--------|
| [`ActiveLearnerConfig<T>`](/docs/reference/wiki/activelearning/activelearnerconfig/) | Comprehensive configuration for active learning. |

## Helpers & Utilities (1)

| Type | Summary |
|:-----|:--------|
| [`InMemoryDatasetFactory<T, TInput, TOutput>`](/docs/reference/wiki/activelearning/inmemorydatasetfactory/) | Factory implementation for creating in-memory datasets. |

