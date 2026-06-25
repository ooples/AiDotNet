---
title: "Evaluation"
description: "All 179 public types in the AiDotNet.evaluation namespace, organized by kind."
section: "API Reference"
---

**179** public types in this namespace, organized by kind.

## Models & Types (125)

| Type | Summary |
|:-----|:--------|
| [`AUCPRMetric<T>`](/docs/reference/wiki/evaluation/aucprmetric/) | Computes Area Under the Precision-Recall Curve (AUC-PR). |
| [`AUCROCMetric<T>`](/docs/reference/wiki/evaluation/aucrocmetric/) | Computes Area Under the ROC Curve (AUC-ROC): measures discrimination ability. |
| [`AccuracyMetric<T>`](/docs/reference/wiki/evaluation/accuracymetric/) | Computes classification accuracy: the proportion of correct predictions. |
| [`AdjustedR2Metric<T>`](/docs/reference/wiki/evaluation/adjustedr2metric/) | Computes Adjusted R² Score: R² adjusted for the number of predictors in the model. |
| [`BalancedAccuracyMetric<T>`](/docs/reference/wiki/evaluation/balancedaccuracymetric/) | Computes balanced accuracy: the average recall across all classes. |
| [`BalancedErrorRateMetric<T>`](/docs/reference/wiki/evaluation/balancederrorratemetric/) | Computes Balanced Error Rate (BER): average of FNR and FPR. |
| [`BlockedKFoldStrategy<T>`](/docs/reference/wiki/evaluation/blockedkfoldstrategy/) | Blocked K-Fold: K-Fold with temporal blocking (gap) between train and validation. |
| [`BootstrapStrategy<T>`](/docs/reference/wiki/evaluation/bootstrapstrategy/) | Bootstrap Cross-Validation: uses bootstrap sampling (sampling with replacement) for validation. |
| [`BootstrapTest<T>`](/docs/reference/wiki/evaluation/bootstraptest/) | Bootstrap-based hypothesis test for comparing two models. |
| [`BrierScoreMetric<T>`](/docs/reference/wiki/evaluation/brierscoremetric/) | Computes Brier Score: mean squared error of probability predictions. |
| [`CRPSMetric<T>`](/docs/reference/wiki/evaluation/crpsmetric/) | Continuous Ranked Probability Score metric for probabilistic predictions. |
| [`CalibrationBin<T>`](/docs/reference/wiki/evaluation/calibrationbin/) | A single bin in the calibration curve. |
| [`CalibrationEngine<T>`](/docs/reference/wiki/evaluation/calibrationengine/) | Engine for analyzing and improving probability calibration of classifiers. |
| [`CalibrationResult<T>`](/docs/reference/wiki/evaluation/calibrationresult/) | Results from calibration analysis. |
| [`ClassificationResults<T>`](/docs/reference/wiki/evaluation/classificationresults/) | Classification-specific results container. |
| [`CochranQTest<T>`](/docs/reference/wiki/evaluation/cochranqtest/) | Cochran's Q test for comparing multiple classifiers on the same dataset. |
| [`CohensKappaMetric<T>`](/docs/reference/wiki/evaluation/cohenskappametric/) | Computes Cohen's Kappa: agreement measure that accounts for chance agreement. |
| [`CrossValidationEngine<T>`](/docs/reference/wiki/evaluation/crossvalidationengine/) | Engine for executing cross-validation with various strategies and aggregating results. |
| [`CrossValidationResult<T>`](/docs/reference/wiki/evaluation/crossvalidationresult/) | Aggregated results from cross-validation across all folds. |
| [`DatasetStatistics<T>`](/docs/reference/wiki/evaluation/datasetstatistics/) | Statistics about the dataset used for evaluation. |
| [`DeLongTest<T>`](/docs/reference/wiki/evaluation/delongtest/) | DeLong's test for comparing two ROC curves. |
| [`DiagnosticOddsRatioMetric<T>`](/docs/reference/wiki/evaluation/diagnosticoddsratiometric/) | Computes Diagnostic Odds Ratio: (TP × TN) / (FP × FN). |
| [`ErrorRateMetric<T>`](/docs/reference/wiki/evaluation/errorratemetric/) | Computes Error Rate (Misclassification Rate): 1 - Accuracy. |
| [`EvaluationReport<T>`](/docs/reference/wiki/evaluation/evaluationreport/) | Comprehensive evaluation report containing all computed metrics and analysis results. |
| [`EvaluationWarning`](/docs/reference/wiki/evaluation/evaluationwarning/) | Warning generated during evaluation. |
| [`ExplainedVarianceMetric<T>`](/docs/reference/wiki/evaluation/explainedvariancemetric/) | Computes Explained Variance Score: proportion of variance in target explained by predictions. |
| [`F1ScoreMetric<T>`](/docs/reference/wiki/evaluation/f1scoremetric/) | Computes F1 score: the harmonic mean of precision and recall. |
| [`FBetaScoreMetric<T>`](/docs/reference/wiki/evaluation/fbetascoremetric/) | Computes F-beta score: weighted harmonic mean of precision and recall. |
| [`FairnessEngine<T>`](/docs/reference/wiki/evaluation/fairnessengine/) | Engine for analyzing fairness and bias in machine learning models. |
| [`FairnessResult<T>`](/docs/reference/wiki/evaluation/fairnessresult/) | Results from fairness analysis. |
| [`FalseDiscoveryRateMetric<T>`](/docs/reference/wiki/evaluation/falsediscoveryratemetric/) | Computes False Discovery Rate (FDR): FP / (FP + TP) = 1 - Precision. |
| [`FalseNegativeRateMetric<T>`](/docs/reference/wiki/evaluation/falsenegativeratemetric/) | Computes False Negative Rate (FNR): proportion of actual positives incorrectly predicted as negative. |
| [`FalseOmissionRateMetric<T>`](/docs/reference/wiki/evaluation/falseomissionratemetric/) | Computes False Omission Rate (FOR): FN / (FN + TN) = 1 - NPV. |
| [`FalsePositiveRateMetric<T>`](/docs/reference/wiki/evaluation/falsepositiveratemetric/) | Computes False Positive Rate (FPR): proportion of actual negatives incorrectly predicted as positive. |
| [`FeatureStatistics<T>`](/docs/reference/wiki/evaluation/featurestatistics/) | Statistics for a single feature. |
| [`FoldResult<T>`](/docs/reference/wiki/evaluation/foldresult/) | Results from a single cross-validation fold. |
| [`FowlkesMallowsMetric<T>`](/docs/reference/wiki/evaluation/fowlkesmallowsmetric/) | Computes the Fowlkes-Mallows Index: geometric mean of precision and recall. |
| [`FriedmanTest<T>`](/docs/reference/wiki/evaluation/friedmantest/) | Friedman test: non-parametric test for comparing multiple classifiers across multiple datasets. |
| [`GiniCoefficientMetric<T>`](/docs/reference/wiki/evaluation/ginicoefficientmetric/) | Computes Gini Coefficient (normalized): 2 * AUC - 1, measures discriminative ability. |
| [`GroupFairnessMetrics`](/docs/reference/wiki/evaluation/groupfairnessmetrics/) | Fairness metrics for a specific group. |
| [`GroupKFoldStrategy<T>`](/docs/reference/wiki/evaluation/groupkfoldstrategy/) | Group K-Fold: K-Fold that keeps related samples (groups) together in the same fold. |
| [`HammingLossMetric<T>`](/docs/reference/wiki/evaluation/hamminglossmetric/) | Computes Hamming Loss: fraction of labels that are incorrectly predicted. |
| [`HingeLossMetric<T>`](/docs/reference/wiki/evaluation/hingelossmetric/) | Computes Hinge Loss: loss function used by Support Vector Machines (SVM). |
| [`HuberLossMetric<T>`](/docs/reference/wiki/evaluation/huberlossmetric/) | Computes Huber Loss: a robust loss function that is less sensitive to outliers than MSE. |
| [`InformednessMetric<T>`](/docs/reference/wiki/evaluation/informednessmetric/) | Computes Informedness (Youden's J statistic): TPR + TNR - 1 = Recall + Specificity - 1. |
| [`JaccardScoreMetric<T>`](/docs/reference/wiki/evaluation/jaccardscoremetric/) | Computes Jaccard Score (Intersection over Union): measures similarity between prediction and actual sets. |
| [`KFoldStrategy<T>`](/docs/reference/wiki/evaluation/kfoldstrategy/) | K-Fold cross-validation: splits data into K equal-sized folds, using each as validation once. |
| [`KruskalWallisTest<T>`](/docs/reference/wiki/evaluation/kruskalwallistest/) | Kruskal-Wallis H test for comparing multiple independent groups. |
| [`LearningCurveEngine<T>`](/docs/reference/wiki/evaluation/learningcurveengine/) | Engine for generating learning curves: how model performance changes with training set size. |
| [`LearningCurveResult<T>`](/docs/reference/wiki/evaluation/learningcurveresult/) | Results from learning curve analysis. |
| [`LeaveOneOutStrategy<T>`](/docs/reference/wiki/evaluation/leaveoneoutstrategy/) | Leave-One-Out Cross-Validation (LOOCV): each sample is used once as validation. |
| [`LeavePOutStrategy<T>`](/docs/reference/wiki/evaluation/leavepoutstrategy/) | Leave-P-Out cross-validation: train on N-P samples, validate on P samples for all combinations. |
| [`LeveneTest<T>`](/docs/reference/wiki/evaluation/levenetest/) | Levene's test for equality of variances across groups. |
| [`LogCoshLossMetric<T>`](/docs/reference/wiki/evaluation/logcoshlossmetric/) | Computes Log-Cosh Loss: mean of log(cosh(y - ŷ)). |
| [`LogLossMetric<T>`](/docs/reference/wiki/evaluation/loglossmetric/) | Computes Log Loss (Cross-Entropy Loss): a probabilistic measure that penalizes confident wrong predictions. |
| [`LogScoreMetric<T>`](/docs/reference/wiki/evaluation/logscoremetric/) | Logarithmic scoring metric for probabilistic predictions (Negative Log Likelihood). |
| [`MAEMetric<T>`](/docs/reference/wiki/evaluation/maemetric/) | Computes Mean Absolute Error (MAE): average absolute difference between predictions and actuals. |
| [`MAPEMetric<T>`](/docs/reference/wiki/evaluation/mapemetric/) | Computes Mean Absolute Percentage Error (MAPE): average percentage error. |
| [`MASEMetric<T>`](/docs/reference/wiki/evaluation/masemetric/) | Computes Mean Absolute Scaled Error (MASE): scale-independent measure for time series. |
| [`MASESeasonalMetric<T>`](/docs/reference/wiki/evaluation/maseseasonalmetric/) | Computes Seasonal MASE: Mean Absolute Scaled Error with explicit seasonal comparison. |
| [`MSEMetric<T>`](/docs/reference/wiki/evaluation/msemetric/) | Computes Mean Squared Error (MSE): average squared difference between predictions and actuals. |
| [`MarkednessMetric<T>`](/docs/reference/wiki/evaluation/markednessmetric/) | Computes Markedness: PPV + NPV - 1 = Precision + NPV - 1. |
| [`MatthewsCorrelationCoefficientMetric<T>`](/docs/reference/wiki/evaluation/matthewscorrelationcoefficientmetric/) |  |
| [`MaxErrorMetric<T>`](/docs/reference/wiki/evaluation/maxerrormetric/) | Computes Maximum Error: the worst-case absolute prediction error. |
| [`McNemarTest<T>`](/docs/reference/wiki/evaluation/mcnemartest/) | McNemar's test: compares the performance of two binary classifiers on the same dataset. |
| [`MeanAbsolutePercentageErrorMetric<T>`](/docs/reference/wiki/evaluation/meanabsolutepercentageerrormetric/) | Computes Mean Absolute Percentage Error with configurable handling of zeros. |
| [`MeanBiasErrorMetric<T>`](/docs/reference/wiki/evaluation/meanbiaserrormetric/) | Computes Mean Bias Error (MBE): average signed error showing systematic over/under-prediction. |
| [`MeanDirectionalAccuracyMetric<T>`](/docs/reference/wiki/evaluation/meandirectionalaccuracymetric/) | Computes Mean Directional Accuracy (MDA): fraction of correctly predicted directions. |
| [`MeanSquaredLogErrorMetric<T>`](/docs/reference/wiki/evaluation/meansquaredlogerrormetric/) | Computes Mean Squared Logarithmic Error (MSLE): squared version of RMSLE. |
| [`MedianAbsoluteErrorMetric<T>`](/docs/reference/wiki/evaluation/medianabsoluteerrormetric/) | Computes Median Absolute Error (MedAE): the median of all absolute errors. |
| [`MetricCollection<T>`](/docs/reference/wiki/evaluation/metriccollection/) | A collection of metrics, providing easy access by name and category. |
| [`MetricComputationEngine<T>`](/docs/reference/wiki/evaluation/metriccomputationengine/) | Core engine for computing evaluation metrics across all task types. |
| [`MetricWithCI<T>`](/docs/reference/wiki/evaluation/metricwithci/) | Represents a metric value with optional confidence interval and metadata. |
| [`ModelComparisonReport<T>`](/docs/reference/wiki/evaluation/modelcomparisonreport/) | Comprehensive report from model comparison. |
| [`MonteCarloStrategy<T>`](/docs/reference/wiki/evaluation/montecarlostrategy/) | Monte Carlo cross-validation (repeated random sub-sampling validation). |
| [`NPVMetric<T>`](/docs/reference/wiki/evaluation/npvmetric/) | Computes Negative Predictive Value (NPV): proportion of negative predictions that are correct. |
| [`NegativeLikelihoodRatioMetric<T>`](/docs/reference/wiki/evaluation/negativelikelihoodratiometric/) | Computes Negative Likelihood Ratio (LR-): (1 - Sensitivity) / Specificity = FNR / TNR. |
| [`NemenyiPostHocTest<T>`](/docs/reference/wiki/evaluation/nemenyiposthoctest/) | Nemenyi post-hoc test: pairwise comparisons after Friedman test. |
| [`NemenyiResult<T>`](/docs/reference/wiki/evaluation/nemenyiresult/) | Results from Nemenyi post-hoc test. |
| [`NestedCVStrategy<T>`](/docs/reference/wiki/evaluation/nestedcvstrategy/) | Nested Cross-Validation: uses an inner CV loop for hyperparameter tuning and outer loop for evaluation. |
| [`NormalizedMSEMetric<T>`](/docs/reference/wiki/evaluation/normalizedmsemetric/) | Computes Normalized Mean Squared Error (NMSE): MSE divided by variance of actuals. |
| [`OptimizedPrecisionMetric<T>`](/docs/reference/wiki/evaluation/optimizedprecisionmetric/) | Computes Optimized Precision: Accuracy - \|Sensitivity - Specificity\| / (Sensitivity + Specificity). |
| [`PairedTTest<T>`](/docs/reference/wiki/evaluation/pairedttest/) | Paired t-test: compares means of two related samples. |
| [`PearsonCorrelationMetric<T>`](/docs/reference/wiki/evaluation/pearsoncorrelationmetric/) | Computes Pearson Correlation Coefficient between predictions and actuals. |
| [`PoissonDevianceMetric<T>`](/docs/reference/wiki/evaluation/poissondeviancemetric/) | Computes Poisson Deviance for count data regression. |
| [`PositiveLikelihoodRatioMetric<T>`](/docs/reference/wiki/evaluation/positivelikelihoodratiometric/) | Computes Positive Likelihood Ratio (LR+): Sensitivity / (1 - Specificity) = TPR / FPR. |
| [`PositivePredictiveValueMetric<T>`](/docs/reference/wiki/evaluation/positivepredictivevaluemetric/) | Computes Positive Predictive Value (PPV): same as Precision but named differently in medical contexts. |
| [`PrecisionMetric<T>`](/docs/reference/wiki/evaluation/precisionmetric/) | Computes precision (positive predictive value): the proportion of positive predictions that are correct. |
| [`PrevalenceMetric<T>`](/docs/reference/wiki/evaluation/prevalencemetric/) | Computes Prevalence: fraction of actual positives in the dataset. |
| [`PrevalenceThresholdMetric<T>`](/docs/reference/wiki/evaluation/prevalencethresholdmetric/) | Computes Prevalence Threshold: the prevalence at which the test would have 50% PPV. |
| [`ProbabilityCalibrator<T>`](/docs/reference/wiki/evaluation/probabilitycalibrator/) | Calibrates model outputs to produce reliable probability estimates. |
| [`PurgedKFoldStrategy<T>`](/docs/reference/wiki/evaluation/purgedkfoldstrategy/) | Purged K-Fold: K-Fold with temporal purging to prevent data leakage in financial/time-dependent data. |
| [`QuantileLossMetric<T>`](/docs/reference/wiki/evaluation/quantilelossmetric/) | Computes Quantile Loss (Pinball Loss) for quantile regression. |
| [`R2ScoreMetric<T>`](/docs/reference/wiki/evaluation/r2scoremetric/) | Computes R² (coefficient of determination): proportion of variance explained by the model. |
| [`RMSEMetric<T>`](/docs/reference/wiki/evaluation/rmsemetric/) | Computes Root Mean Squared Error (RMSE): square root of MSE. |
| [`RMSLEMetric<T>`](/docs/reference/wiki/evaluation/rmslemetric/) | Computes Root Mean Squared Logarithmic Error (RMSLE): measures ratio errors rather than absolute errors. |
| [`RecallMetric<T>`](/docs/reference/wiki/evaluation/recallmetric/) | Computes recall (sensitivity, true positive rate): the proportion of actual positives correctly identified. |
| [`RegressionResults<T>`](/docs/reference/wiki/evaluation/regressionresults/) | Regression-specific results container. |
| [`RelativeAbsoluteErrorMetric<T>`](/docs/reference/wiki/evaluation/relativeabsoluteerrormetric/) | Computes Relative Absolute Error (RAE): sum of absolute errors relative to baseline. |
| [`RelativeSquaredErrorMetric<T>`](/docs/reference/wiki/evaluation/relativesquarederrormetric/) | Computes Relative Squared Error (RSE): sum of squared errors relative to baseline model. |
| [`RepeatedKFoldStrategy<T>`](/docs/reference/wiki/evaluation/repeatedkfoldstrategy/) | Repeated K-Fold: runs K-Fold multiple times with different random shuffles. |
| [`ResidualStatistics<T>`](/docs/reference/wiki/evaluation/residualstatistics/) | Statistics about regression residuals. |
| [`RobustnessEngine<T>`](/docs/reference/wiki/evaluation/robustnessengine/) | Engine for analyzing model robustness to input perturbations and noise. |
| [`RobustnessResult<T>`](/docs/reference/wiki/evaluation/robustnessresult/) | Results from robustness analysis. |
| [`SMAPEMetric<T>`](/docs/reference/wiki/evaluation/smapemetric/) | Computes Symmetric Mean Absolute Percentage Error (SMAPE): a bounded percentage error metric. |
| [`ShuffleSplitStrategy<T>`](/docs/reference/wiki/evaluation/shufflesplitstrategy/) | Shuffle Split (Monte Carlo Cross-Validation): random train/test splits repeated multiple times. |
| [`SlidingWindowStrategy<T>`](/docs/reference/wiki/evaluation/slidingwindowstrategy/) | Sliding Window cross-validation for time series with fixed-size training window. |
| [`SpearmanCorrelationMetric<T>`](/docs/reference/wiki/evaluation/spearmancorrelationmetric/) | Computes Spearman's Rank Correlation Coefficient between predictions and actuals. |
| [`SpecificityMetric<T>`](/docs/reference/wiki/evaluation/specificitymetric/) | Computes specificity (true negative rate): the proportion of actual negatives correctly identified. |
| [`StatisticalTestEngine<T>`](/docs/reference/wiki/evaluation/statisticaltestengine/) | Engine for performing statistical tests on model comparison results. |
| [`StatisticalTestResult<T>`](/docs/reference/wiki/evaluation/statisticaltestresult/) | Represents the result of a statistical test. |
| [`StratifiedGroupKFoldStrategy<T>`](/docs/reference/wiki/evaluation/stratifiedgroupkfoldstrategy/) | Stratified Group K-Fold: combines stratification and group constraints. |
| [`StratifiedKFoldStrategy<T>`](/docs/reference/wiki/evaluation/stratifiedkfoldstrategy/) | Stratified K-Fold: K-Fold that preserves the percentage of samples for each class. |
| [`SymmetricMAPEMetric<T>`](/docs/reference/wiki/evaluation/symmetricmapemetric/) | Computes Symmetric Mean Absolute Percentage Error (sMAPE) for regression. |
| [`TheilUMetric<T>`](/docs/reference/wiki/evaluation/theilumetric/) | Computes Theil's U Statistic: measures forecast accuracy relative to a naive no-change forecast. |
| [`ThreatScoreMetric<T>`](/docs/reference/wiki/evaluation/threatscoremetric/) | Computes Threat Score (Critical Success Index): TP / (TP + FN + FP). |
| [`TimeSeriesSplitStrategy<T>`](/docs/reference/wiki/evaluation/timeseriessplitstrategy/) | Time Series Split: expanding window cross-validation that respects temporal order. |
| [`TrueNegativeRateMetric<T>`](/docs/reference/wiki/evaluation/truenegativeratemetric/) | Computes True Negative Rate (TNR): same as Specificity, TN / (TN + FP). |
| [`TweedieLossMetric<T>`](/docs/reference/wiki/evaluation/tweedielossmetric/) | Computes Tweedie Deviance Loss for regression with power parameter. |
| [`ValidationCurveEngine<T>`](/docs/reference/wiki/evaluation/validationcurveengine/) | Engine for generating validation curves: how model performance changes with hyperparameter values. |
| [`ValidationCurveResult<T>`](/docs/reference/wiki/evaluation/validationcurveresult/) | Results from validation curve analysis. |
| [`WAPEMetric<T>`](/docs/reference/wiki/evaluation/wapemetric/) | Computes Weighted Absolute Percentage Error (WAPE): total absolute error as percentage of total actuals. |
| [`WeightedMAPEMetric<T>`](/docs/reference/wiki/evaluation/weightedmapemetric/) | Computes Weighted Mean Absolute Percentage Error (wMAPE): weighted by actual values. |
| [`WilcoxonSignedRankTest<T>`](/docs/reference/wiki/evaluation/wilcoxonsignedranktest/) | Wilcoxon signed-rank test: non-parametric paired comparison test. |
| [`ZeroOneLossMetric<T>`](/docs/reference/wiki/evaluation/zeroonelossmetric/) | Computes Zero-One Loss: fraction of misclassifications (complement of accuracy). |

## Interfaces (13)

| Type | Summary |
|:-----|:--------|
| [`IClassificationMetric<T>`](/docs/reference/wiki/evaluation/iclassificationmetric/) | Interface for classification metrics. |
| [`IClassifierComparisonTest<T>`](/docs/reference/wiki/evaluation/iclassifiercomparisontest/) | Interface for tests comparing multiple classifier predictions against ground truth. |
| [`ICrossValidationStrategy<T>`](/docs/reference/wiki/evaluation/icrossvalidationstrategy/) | Defines a cross-validation splitting strategy. |
| [`IMetric<T>`](/docs/reference/wiki/evaluation/imetric/) | Base interface for all evaluation metrics. |
| [`IMultipleComparisonTest<T>`](/docs/reference/wiki/evaluation/imultiplecomparisontest/) | Interface for tests comparing multiple classifiers or groups. |
| [`IMultipleSampleTest<T>`](/docs/reference/wiki/evaluation/imultiplesampletest/) | Interface for statistical tests comparing multiple groups. |
| [`IPairedTest<T>`](/docs/reference/wiki/evaluation/ipairedtest/) | Interface for paired comparison tests (e.g., comparing same samples under different conditions). |
| [`IProbabilisticClassificationMetric<T>`](/docs/reference/wiki/evaluation/iprobabilisticclassificationmetric/) | Interface for classification metrics that use probabilities. |
| [`IRankingMetric<T>`](/docs/reference/wiki/evaluation/irankingmetric/) | Interface for ranking/recommendation metrics. |
| [`IRegressionMetric<T>`](/docs/reference/wiki/evaluation/iregressionmetric/) | Interface for regression metrics. |
| [`IStatisticalTest<T>`](/docs/reference/wiki/evaluation/istatisticaltest/) | Interface for general statistical tests. |
| [`ITimeSeriesMetric<T>`](/docs/reference/wiki/evaluation/itimeseriesmetric/) | Interface for time series forecasting metrics. |
| [`ITwoSampleTest<T>`](/docs/reference/wiki/evaluation/itwosampletest/) | Interface for statistical tests comparing two groups or samples. |

## Enums (27)

| Type | Summary |
|:-----|:--------|
| [`AdversarialAttackMethod`](/docs/reference/wiki/evaluation/adversarialattackmethod/) | Methods for generating adversarial examples. |
| [`AveragingMethod`](/docs/reference/wiki/evaluation/averagingmethod/) | Specifies the averaging method for multi-class/multi-label classification metrics. |
| [`BiasVarianceDiagnosis`](/docs/reference/wiki/evaluation/biasvariancediagnosis/) | Specifies the diagnosed bias-variance condition of a model. |
| [`BinningStrategy`](/docs/reference/wiki/evaluation/binningstrategy/) | Strategies for binning predictions in calibration analysis. |
| [`CenterType<T>`](/docs/reference/wiki/evaluation/centertype/) |  |
| [`ConfidenceIntervalMethod`](/docs/reference/wiki/evaluation/confidenceintervalmethod/) | Specifies the method for computing confidence intervals on evaluation metrics. |
| [`ContinuousBinningStrategy`](/docs/reference/wiki/evaluation/continuousbinningstrategy/) | Strategy for binning continuous features. |
| [`CrossValidationStrategy`](/docs/reference/wiki/evaluation/crossvalidationstrategy/) | Specifies the cross-validation strategy to use for model evaluation. |
| [`EffectSizeMeasure`](/docs/reference/wiki/evaluation/effectsizemeasure/) | Effect size measures for comparing models. |
| [`FairnessConstraint`](/docs/reference/wiki/evaluation/fairnessconstraint/) | Specifies the fairness constraint or metric for model evaluation. |
| [`MetricDirection`](/docs/reference/wiki/evaluation/metricdirection/) | Specifies whether higher or lower values are better for a metric. |
| [`MissingValueStrategy`](/docs/reference/wiki/evaluation/missingvaluestrategy/) | Strategies for handling missing values in robustness testing. |
| [`MultipleComparisonTest`](/docs/reference/wiki/evaluation/multiplecomparisontest/) | Tests for comparing multiple models simultaneously. |
| [`MultipleTestingCorrectionMethod`](/docs/reference/wiki/evaluation/multipletestingcorrectionmethod/) | Methods for correcting p-values when performing multiple comparisons. |
| [`NormalizationType`](/docs/reference/wiki/evaluation/normalizationtype/) | Specifies the normalization method for confusion matrices and metrics. |
| [`OptimalValueSelectionMethod`](/docs/reference/wiki/evaluation/optimalvalueselectionmethod/) | Method for selecting optimal hyperparameter value from validation curve. |
| [`PairwiseComparisonTest`](/docs/reference/wiki/evaluation/pairwisecomparisontest/) | Tests for pairwise model comparison. |
| [`PostHocTest`](/docs/reference/wiki/evaluation/posthoctest/) | Post-hoc tests after significant omnibus test. |
| [`PredictionIntervalMethod`](/docs/reference/wiki/evaluation/predictionintervalmethod/) | Methods for computing prediction intervals. |
| [`ReportDetailLevel`](/docs/reference/wiki/evaluation/reportdetaillevel/) | Detail level for evaluation reports. |
| [`ReportFormat`](/docs/reference/wiki/evaluation/reportformat/) | Specifies the output format for evaluation reports. |
| [`ReportSection`](/docs/reference/wiki/evaluation/reportsection/) | Sections that can be included in evaluation reports. |
| [`SliceSortOrder`](/docs/reference/wiki/evaluation/slicesortorder/) | Sort order for slices in reports. |
| [`ThresholdSelectionMethod`](/docs/reference/wiki/evaluation/thresholdselectionmethod/) | Specifies the method for selecting classification thresholds. |
| [`UncertaintyCalibrationMethod`](/docs/reference/wiki/evaluation/uncertaintycalibrationmethod/) | Methods for calibrating uncertainty estimates. |
| [`UncertaintyDecompositionMethod`](/docs/reference/wiki/evaluation/uncertaintydecompositionmethod/) | Methods for decomposing predictive uncertainty into components. |
| [`WarningSeverity`](/docs/reference/wiki/evaluation/warningseverity/) | Severity levels for evaluation warnings. |

## Structs (1)

| Type | Summary |
|:-----|:--------|
| [`CVFold<T>`](/docs/reference/wiki/evaluation/cvfold/) | Represents a single cross-validation fold with train and validation data. |

## Options & Configuration (13)

| Type | Summary |
|:-----|:--------|
| [`CalibrationOptions`](/docs/reference/wiki/evaluation/calibrationoptions/) | Configuration options for probability calibration analysis. |
| [`CrossValidationOptions`](/docs/reference/wiki/evaluation/crossvalidationoptions/) | Configuration options for cross-validation. |
| [`EvaluationOptions<T>`](/docs/reference/wiki/evaluation/evaluationoptions/) | Configuration options for model evaluation. |
| [`FairnessOptions`](/docs/reference/wiki/evaluation/fairnessoptions/) | Configuration options for fairness evaluation. |
| [`InfluenceAnalysisOptions`](/docs/reference/wiki/evaluation/influenceanalysisoptions/) | Configuration options for influence analysis in regression models. |
| [`LearningCurveOptions`](/docs/reference/wiki/evaluation/learningcurveoptions/) | Configuration options for learning curve analysis. |
| [`ModelComparisonOptions`](/docs/reference/wiki/evaluation/modelcomparisonoptions/) | Configuration options for comparing multiple models. |
| [`ReportOptions`](/docs/reference/wiki/evaluation/reportoptions/) | Configuration options for evaluation report generation. |
| [`ResidualAnalysisOptions`](/docs/reference/wiki/evaluation/residualanalysisoptions/) | Configuration options for residual analysis in regression models. |
| [`RobustnessOptions`](/docs/reference/wiki/evaluation/robustnessoptions/) | Configuration options for robustness evaluation. |
| [`SubgroupAnalysisOptions`](/docs/reference/wiki/evaluation/subgroupanalysisoptions/) | Configuration options for subgroup (slice-based) analysis. |
| [`UncertaintyOptions`](/docs/reference/wiki/evaluation/uncertaintyoptions/) | Configuration options for uncertainty quantification. |
| [`ValidationCurveOptions`](/docs/reference/wiki/evaluation/validationcurveoptions/) | Configuration options for validation curve analysis. |

