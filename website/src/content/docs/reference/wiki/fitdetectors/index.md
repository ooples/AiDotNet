---
title: "Fit Detectors"
description: "All 32 public types in the AiDotNet.fitdetectors namespace, organized by kind."
section: "API Reference"
---

**32** public types in this namespace, organized by kind.

## Models & Types (31)

| Type | Summary |
|:-----|:--------|
| [`AdaptiveFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/adaptivefitdetector/) | An adaptive fit detector that dynamically selects the most appropriate detection method based on data characteristics. |
| [`AutocorrelationFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/autocorrelationfitdetector/) | A fit detector that analyzes autocorrelation in model residuals to assess model fit. |
| [`BayesianFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/bayesianfitdetector/) | A fit detector that uses Bayesian model comparison metrics to assess model fit. |
| [`BootstrapFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/bootstrapfitdetector/) | A fit detector that uses bootstrap resampling to assess model fit and stability. |
| [`CalibratedProbabilityFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/calibratedprobabilityfitdetector/) | A fit detector that analyzes the calibration of probability predictions to assess model fit. |
| [`ConfusionMatrixFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/confusionmatrixfitdetector/) | A fit detector that analyzes confusion matrix metrics to assess classification model fit. |
| [`CookDistanceFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/cookdistancefitdetector/) | A fit detector that uses Cook's distance to identify influential data points and assess model fit. |
| [`CrossValidationFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/crossvalidationfitdetector/) | A fit detector that analyzes model performance across training, validation, and test datasets to assess model fit. |
| [`DefaultFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/defaultfitdetector/) | A default implementation of a fit detector that analyzes model performance and provides recommendations. |
| [`EnsembleFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/ensemblefitdetector/) | A fit detector that combines the results of multiple individual fit detectors to provide a more robust assessment. |
| [`FeatureImportanceFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/featureimportancefitdetector/) | A fit detector that analyzes feature importances and correlations to assess model fit. |
| [`GaussianProcessFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/gaussianprocessfitdetector/) | A fit detector that uses Gaussian Process regression to analyze model uncertainty and performance. |
| [`GradientBoostingFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/gradientboostingfitdetector/) | A specialized detector that evaluates how well a gradient boosting model fits the data. |
| [`HeteroscedasticityFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/heteroscedasticityfitdetector/) | A detector that evaluates whether a model's errors have consistent variance across all predictions. |
| [`HoldoutValidationFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/holdoutvalidationfitdetector/) | A detector that evaluates model fit quality using holdout validation techniques. |
| [`HybridFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/hybridfitdetector/) | A detector that combines multiple fit detection approaches to provide more robust model evaluation. |
| [`InformationCriteriaFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/informationcriteriafitdetector/) | A detector that evaluates model fit using information criteria metrics (AIC and BIC). |
| [`JackknifeFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/jackknifefitdetector/) | A detector that evaluates model fit using the jackknife resampling technique. |
| [`KFoldCrossValidationFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/kfoldcrossvalidationfitdetector/) | A detector that evaluates model fit using K-Fold Cross-Validation technique. |
| [`LearningCurveFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/learningcurvefitdetector/) | A detector that evaluates model fit by analyzing learning curves from training and validation data. |
| [`NeuralNetworkFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/neuralnetworkfitdetector/) | A specialized detector for evaluating the fit quality of neural network models. |
| [`PartialDependencePlotFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/partialdependenceplotfitdetector/) | A fit detector that uses Partial Dependence Plots to analyze model fit and detect overfitting or underfitting. |
| [`PermutationTestFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/permutationtestfitdetector/) | A detector that uses permutation testing to evaluate model fit quality. |
| [`PrecisionRecallCurveFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/precisionrecallcurvefitdetector/) | A detector that evaluates model fit quality using precision-recall curve metrics. |
| [`ROCCurveFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/roccurvefitdetector/) | A detector that evaluates model fit quality using ROC (Receiver Operating Characteristic) curve analysis. |
| [`ResidualAnalysisFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/residualanalysisfitdetector/) | A detector that evaluates model fit quality by analyzing the residuals (errors) of the model. |
| [`ResidualBootstrapFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/residualbootstrapfitdetector/) | A detector that evaluates model fit quality using residual bootstrap resampling techniques. |
| [`ShapleyValueFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/shapleyvaluefitdetector/) | A detector that evaluates model fit quality using Shapley values to determine feature importance. |
| [`StratifiedKFoldCrossValidationFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/stratifiedkfoldcrossvalidationfitdetector/) | A detector that evaluates model fit using Stratified K-Fold Cross-Validation. |
| [`TimeSeriesCrossValidationFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/timeseriescrossvalidationfitdetector/) | A specialized detector that evaluates how well a model fits time series data using cross-validation techniques. |
| [`VIFFitDetector<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/viffitdetector/) | A detector that evaluates model fit by analyzing Variance Inflation Factor (VIF) to identify multicollinearity issues. |

## Base Classes (1)

| Type | Summary |
|:-----|:--------|
| [`FitDetectorBase<T, TInput, TOutput>`](/docs/reference/wiki/fitdetectors/fitdetectorbase/) | Base class for all fit detectors that provides common functionality and defines the required interface. |

