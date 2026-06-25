---
title: "MetricType"
description: "Defines the types of metrics used to evaluate machine learning models."
section: "API Reference"
---

`Enums` · `AiDotNet.Enums`

Defines the types of metrics used to evaluate machine learning models.

## Fields

| Field | Summary |
|:-----|:--------|
| `AIC` | A measure of the relative quality of statistical models for a given set of data. |
| `AICAlt` | Alternative Akaike Information Criterion - A variant of AIC with a different penalty term. |
| `AUC` | Alias of `AUCROC`. |
| `AUCPR` | Area Under the Precision-Recall Curve - Measures classification accuracy focusing on positive cases. |
| `AUCROC` | Area Under the Receiver Operating Characteristic Curve - Measures classification accuracy across thresholds. |
| `Accuracy` | The proportion of correct predictions among all predictions made. |
| `AdjustedR2` | A modified version of R² that accounts for the number of predictors in the model. |
| `AdjustedRandIndex` | Measures the similarity between two clusterings adjusted for chance. |
| `AverageEpisodeReward` | The average total reward per episode (reinforcement learning). |
| `AveragePrecision` | Measures the average precision across different recall levels. |
| `BIC` | A criterion for model selection that penalizes model complexity more strongly than AIC. |
| `BhattacharyyaDistance` | Measures the similarity between two probability distributions. |
| `BrierScore` | A proper score function that measures the accuracy of probabilistic predictions. |
| `CRPS` | Continuous Ranked Probability Score for probabilistic forecasts. |
| `CalibrationError` | Measures how well the predicted probabilities match the actual outcomes. |
| `CalinskiHarabaszIndex` | A measure of how similar an object is to its own cluster compared to other clusters. |
| `ChamferDistance` | Chamfer Distance for point cloud similarity measurement. |
| `CohenKappa` | Measures the agreement between two sets of labels. |
| `ConditionNumber` | Measures the sensitivity of a system to small changes in input. |
| `CosineSimilarity` | A measure of similarity between two non-zero vectors. |
| `CrossEntropyLoss` | Measures the average log-loss across all classes in classification problems. |
| `DaviesBouldinIndex` | A metric for evaluating clustering algorithms. |
| `DurbinWatsonStatistic` | Durbin-Watson Statistic - Detects autocorrelation in prediction errors. |
| `EarthMoversDistance` | Earth Mover's Distance (Wasserstein) for point cloud comparison. |
| `EffectiveNumberOfParameters` | The effective number of parameters in a model, useful for model complexity assessment. |
| `EuclideanDistance` | The straight-line distance between two points in Euclidean space. |
| `ExplainedVarianceScore` | Measures the proportion of variance in the dependent variable explained by the model. |
| `F1Score` | The harmonic mean of precision and recall, providing a balance between the two metrics. |
| `F2Score` | Measures the harmonic mean of precision and recall, with more weight on recall. |
| `FirstQuartile` | The first quartile (25th percentile) of the dataset. |
| `GMean` | Measures the geometric mean of precision and recall. |
| `HammingDistance` | The number of positions at which the corresponding symbols are different. |
| `InterquartileRange` | The range of the middle 50% of the data. |
| `IoU3D` | Intersection over Union for 3D volumetric segmentation. |
| `JaccardSimilarity` | A statistic used for comparing the similarity and diversity of sample sets. |
| `KLDivergence` | Measures the difference between two probability distributions. |
| `KendallTau` | Measures the ordinal association between predicted and actual values. |
| `LPIPS` | Learned Perceptual Image Patch Similarity for deep perceptual quality. |
| `LevenshteinDistance` | Measures the similarity between two sequences. |
| `Likelihood` | A measure of how probable the observed data is under the model. |
| `LogLikelihood` | The natural logarithm of the likelihood, used for numerical stability and easier interpretation. |
| `LogPointwisePredictiveDensity` | A measure of the model's predictive accuracy on a point-by-point basis. |
| `MAD` | The median absolute deviation (MAD) of the dataset. |
| `MAE` | Mean Absolute Error - The average absolute difference between predicted and actual values. |
| `MAPE` | Mean Absolute Percentage Error - The average percentage difference between predicted and actual values. |
| `MSE` | Mean Squared Error - The average of squared differences between predicted and actual values. |
| `MahalanobisDistance` | A multi-dimensional generalization of measuring how many standard deviations away a point is from the mean of a distribution. |
| `ManhattanDistance` | The sum of the absolute differences of the coordinates. |
| `MarginalLikelihood` | The probability of the observed data under all possible parameter values. |
| `Max` | The largest value in the dataset. |
| `MaxError` | Measures the maximum error between predicted and actual values. |
| `Mean` | The arithmetic average of a set of values. |
| `MeanAbsoluteError` | The average of the absolute differences between predicted and actual values. |
| `MeanAbsolutePercentageError` | The average of the absolute percentage differences between predicted and actual values. |
| `MeanAveragePrecision` | The mean of the average precision scores for each query. |
| `MeanBiasError` | Mean Bias Error - The average of prediction errors (predicted - actual). |
| `MeanIoU` | Mean Intersection over Union across all classes. |
| `MeanPredictionError` | The average difference between predicted values and actual values. |
| `MeanReciprocalRank` | The average of the reciprocal ranks of the first relevant items. |
| `MeanSquaredError` | The average of the squared differences between predicted and actual values. |
| `MeanSquaredLogError` | Mean Squared Logarithmic Error - Penalizes underestimates more than overestimates. |
| `Median` | The middle value in a sorted list of numbers. |
| `MedianAbsoluteError` | Median Absolute Error - The middle value of all absolute differences between predicted and actual values. |
| `MedianPredictionError` | The middle value of all differences between predicted values and actual values. |
| `Min` | The smallest value in the dataset. |
| `Mode` | The most frequently occurring value in a dataset. |
| `MutualInformation` | Measures the mutual dependence between two variables. |
| `N` | The number of values in the dataset. |
| `NormalConsistency` | Normal consistency for surface reconstruction quality. |
| `NormalizedDiscountedCumulativeGain` | A measure of ranking quality used in information retrieval. |
| `NormalizedMutualInformation` | A normalization of the Mutual Information to scale the results between 0 and 1. |
| `ObservedTestStatistic` | The value of the test statistic computed from the observed data. |
| `PSNR` | Peak Signal-to-Noise Ratio for rendered image quality. |
| `PartIoU` | Part-averaged IoU for part segmentation tasks. |
| `PearsonCorrelation` | Measures the linear correlation between predicted and actual values. |
| `Perplexity` | Measures the log-likelihood of held-out data under a topic model. |
| `PopulationStandardError` | Population Standard Error - The standard deviation of prediction errors without adjustment for model complexity. |
| `Precision` | The proportion of true positive predictions among all positive predictions. |
| `PredictionIntervalCoverage` | The percentage of actual values that fall within the model's prediction intervals. |
| `R2` | Coefficient of determination, measuring how well the model explains the variance in the data. |
| `RMSE` | Root Mean Squared Error - The square root of the Mean Squared Error. |
| `ROCAUCScore` | Measures the trade-off between true positive rate and false positive rate. |
| `RSquared` | Alias of `R2`. |
| `Range` | The difference between the largest and smallest values in a dataset. |
| `Recall` | The proportion of true positive predictions among all actual positives. |
| `ReferenceModelMarginalLikelihood` | The marginal likelihood of a reference or null model. |
| `RootMeanSquaredError` | The square root of the mean squared error. |
| `SMAPE` | Symmetric Mean Absolute Percentage Error - A variant of MAPE that handles zero or near-zero values better. |
| `SSIM` | Structural Similarity Index for perceptual image quality. |
| `SampleStandardError` | Sample Standard Error - An estimate of the standard deviation of prediction errors, adjusted for model complexity. |
| `SilhouetteScore` | Measures the quality of clustering algorithms. |
| `SpearmanCorrelation` | Measures the monotonic relationship between predicted and actual values. |
| `StandardDeviation` | The square root of the variance, measuring the average deviation from the mean. |
| `TheilUStatistic` | Theil's U Statistic - A measure of forecast accuracy relative to a naive forecasting method. |
| `ThirdQuartile` | The third quartile (75th percentile) of the dataset. |
| `Variance` | A measure of variability in the data, calculated as the average squared deviation from the mean. |
| `VariationOfInformation` | A measure of the amount of information lost when compressing two random variables. |

