---
title: "StatisticsHelper<T>"
description: "Provides statistical calculation methods for various data analysis tasks."
section: "API Reference"
---

`Helpers & Utilities` · `AiDotNet.Helpers`

Provides statistical calculation methods for various data analysis tasks.

## For Beginners

This class contains methods to calculate common statistical measures 
like averages, variations, and statistical tests. These help you understand your data's 
patterns and make decisions based on statistical evidence.

## Methods

| Method | Summary |
|:-----|:--------|
| `BetaIncompleteContinuedFraction(,,)` | Computes the regularized incomplete beta function using continued fraction (Lentz's algorithm). |
| `BetaPDF(,,)` | Calculates the probability density function (PDF) of the Beta distribution. |
| `CalculateAIC(Int32,Int32,)` | Calculates the Akaike Information Criterion (AIC) for model comparison. |
| `CalculateAICAlternative(Int32,Int32,)` | Calculates an alternative formulation of the Akaike Information Criterion (AIC). |
| `CalculateAUC(Vector<>,Vector<>)` | Calculates the Area Under a Curve (AUC) given x and y coordinates. |
| `CalculateAccuracy(Vector<>,Vector<>)` | Calculates the accuracy of predictions by comparing them to actual values. |
| `CalculateAccuracy(Vector<>,Vector<>,PredictionType,)` | Calculates the accuracy of predictions with support for different prediction types and tolerance levels. |
| `CalculateAdjustedR2(,Int32,Int32)` | Calculates the adjusted R² value, which accounts for the number of predictors in the model. |
| `CalculateAdjustedRandIndex(Vector<>,Vector<>)` | Calculates the Adjusted Rand Index (ARI) between two clusterings. |
| `CalculateAucF1Score(ModelEvaluationData<,,>)` | Calculates both the AUC and F1 score for model evaluation. |
| `CalculateAutoCorrelationFunction(Vector<>,Int32)` | Calculates the autocorrelation function (ACF) for a time series up to a specified maximum lag. |
| `CalculateAverageDistance(Matrix<>,Vector<>,String,Vector<>)` | Calculates the average distance from all points in a cluster to the cluster's centroid. |
| `CalculateAverageDistanceToCluster(Matrix<>,Vector<>,Int32,)` | Calculates the average distance from a point to all points in a specific cluster. |
| `CalculateAverageIntraClusterDistance(Matrix<>,Vector<>,Int32)` | Calculates the average distance from a point to all other points in the same cluster. |
| `CalculateBIC(Int32,Int32,)` | Calculates the Bayesian Information Criterion (BIC) for model comparison. |
| `CalculateBayesFactor(ModelStats<,,>)` | Calculates the Bayes Factor for comparing two models. |
| `CalculateBetaCDF(,,)` | Calculates the cumulative distribution function (CDF) of the Beta distribution. |
| `CalculateBetweenClusterVariance(Dictionary<String,Vector<>>,Vector<>,Vector<>)` | Calculates the between-cluster variance for a clustering result. |
| `CalculateBootstrapInterval(Vector<>,Vector<>,)` | Calculates confidence intervals using bootstrap resampling. |
| `CalculateCRPS(Vector<>,Vector<>)` | Calculates the CRPS for point predictions (assumes zero uncertainty). |
| `CalculateCRPS(Vector<>,Vector<>,Vector<>)` | Calculates the Continuous Ranked Probability Score (CRPS) for probabilistic forecasts. |
| `CalculateCalinskiHarabaszIndex(Matrix<>,Vector<>)` | Calculates the Calinski-Harabasz Index (Variance Ratio Criterion) for a clustering result. |
| `CalculateCentroids(Matrix<>,Vector<>)` | Calculates the centroids for each cluster in a clustering result. |
| `CalculateChiSquareCDF(Int32,)` | Calculates the chi-square cumulative distribution function (CDF) value for a given chi-square value and degrees of freedom. |
| `CalculateChiSquarePDF(Int32,)` | Calculates the probability density function (PDF) of the chi-square distribution. |
| `CalculateClopperPearsonInterval(Int32,Int32,)` | Calculates the Clopper-Pearson (exact) confidence interval for a binomial proportion. |
| `CalculateClopperPearsonLowerBound(Int32,Int32,)` | Calculates the Clopper-Pearson lower confidence bound for a binomial proportion. |
| `CalculateConditionNumber(Matrix<>,ModelStatsOptions)` | Calculates the condition number of a matrix using the specified method. |
| `CalculateConditionNumberL1Norm(Matrix<>)` | Calculates the condition number of a matrix using the L1 norm. |
| `CalculateConditionNumberLInfNorm(Matrix<>)` | Calculates the condition number of a matrix using the infinity norm. |
| `CalculateConditionNumberPowerIteration(Matrix<>,Int32,)` | Calculates the condition number of a matrix using the power iteration method. |
| `CalculateConditionNumberSVD(Matrix<>)` | Calculates the condition number of a matrix using Singular Value Decomposition (SVD). |
| `CalculateConfidenceIntervals(Vector<>,,DistributionType)` | Calculates confidence intervals for the mean of a set of values based on the specified distribution type and confidence level. |
| `CalculateConfusionMatrix(Vector<>,Vector<>,)` | Calculates a confusion matrix for binary classification at a specified threshold. |
| `CalculateCorrelationMatrix(Matrix<>,ModelStatsOptions)` | Calculates a correlation matrix for a set of features. |
| `CalculateCovarianceMatrix(Matrix<>)` | Calculates the covariance matrix for a dataset. |
| `CalculateCredibleIntervals(Vector<>,,DistributionType)` | Calculates the credible intervals for a given set of values based on the specified distribution type and confidence level. |
| `CalculateDIC(ModelStats<,,>)` | Calculates the Deviance Information Criterion (DIC) for Bayesian model comparison. |
| `CalculateDaviesBouldinIndex(Matrix<>,Vector<>)` | Calculates the Davies-Bouldin Index for a clustering result. |
| `CalculateDistance(Vector<>,Vector<>,DistanceMetricType,Matrix<>)` | Calculates the distance or similarity between two vectors using the specified metric. |
| `CalculateDurbinWatsonStatistic(List<>)` | Calculates the Durbin-Watson statistic from a list of residuals. |
| `CalculateDurbinWatsonStatistic(Vector<>,Vector<>)` | Calculates the Durbin-Watson statistic to test for autocorrelation in residuals. |
| `CalculateDynamicTimeWarping(Vector<>,Vector<>)` | Calculates the Dynamic Time Warping (DTW) distance between two time series. |
| `CalculateEffectiveNumberOfParameters(Matrix<>,Vector<>)` | Calculates the effective number of parameters in a model using the trace of the hat matrix. |
| `CalculateEntropy(Vector<>)` | Calculates the Shannon entropy of a discrete random variable. |
| `CalculateExplainedVarianceScore(Vector<>,Vector<>)` | Calculates the explained variance score between actual and predicted values. |
| `CalculateExponentialPDF(,)` | Calculates the probability density function (PDF) value for an exponential distribution. |
| `CalculateF1Score(,)` | Calculates the F1 score from precision and recall values. |
| `CalculateFDistributionQuantile(,Int32,Int32)` | Calculates a quantile (inverse cumulative distribution function) of the F-distribution. |
| `CalculateForecastInterval(Vector<>,Vector<>,)` | Calculates a forecast interval for future predictions. |
| `CalculateFriedmanMSE(Vector<>,List<Int32>,List<Int32>)` | Calculates the Friedman Mean Squared Error for a potential split in a decision tree. |
| `CalculateGlobalCentroid(Matrix<>)` | Calculates the global centroid (mean) of all data points. |
| `CalculateGoodnessOfFit(Vector<>,Func<,>)` | Calculates the goodness of fit for a probability distribution against sample data. |
| `CalculateInverseBetaCDF(,,)` | Calculates the inverse cumulative distribution function (quantile function) of the Beta distribution. |
| `CalculateInverseChiSquareCDF(Int32,)` | Calculates the inverse of the chi-square cumulative distribution function (CDF). |
| `CalculateInverseExponentialCDF(,)` | Calculates the inverse cumulative distribution function (CDF) of the exponential distribution. |
| `CalculateInverseLaplaceCDF(,,)` | Calculates the inverse of the Laplace cumulative distribution function (CDF). |
| `CalculateInverseNormalCDF()` | Calculates the inverse of the standard normal cumulative distribution function (CDF). |
| `CalculateInverseNormalCDF(,,)` | Calculates the inverse of the normal cumulative distribution function (CDF) with specified mean and standard deviation. |
| `CalculateInverseStudentTCDF(Int32,)` | Calculates the inverse of the Student's t cumulative distribution function (CDF). |
| `CalculateJackknifeInterval(Vector<>,Vector<>)` | Calculates confidence intervals using the jackknife resampling method. |
| `CalculateKendallTau(Vector<>,Vector<>)` | Calculates Kendall's tau correlation coefficient between two vectors. |
| `CalculateLOO(ModelStats<,,>)` | Calculates the Leave-One-Out Cross-Validation (LOO-CV) criterion for Bayesian model comparison. |
| `CalculateLaplacePDF(,,)` | Calculates the probability density function (PDF) value for a Laplace distribution. |
| `CalculateLearningCurve(Vector<>,Vector<>,Int32)` | Calculates a learning curve by evaluating model performance on increasingly larger subsets of data. |
| `CalculateLeaveOneOutPredictiveDensities(Matrix<>,Vector<>,Func<Matrix<>,Vector<>,Vector<>>)` | Calculates leave-one-out predictive densities for each observation. |
| `CalculateLikelihood(,)` | Calculates the likelihood of an observed value given a predicted value. |
| `CalculateLogLikelihood(Vector<>,Vector<>)` | Calculates the log-likelihood of a model given actual and predicted values. |
| `CalculateLogNormalPDF(,,)` | Calculates the probability density function (PDF) value for a log-normal distribution. |
| `CalculateLogPointwisePredictiveDensity(Vector<>,Vector<>)` | Calculates the log pointwise predictive density (LPPD) for a model. |
| `CalculateMarginalLikelihood(Vector<>,Vector<>,Int32)` | Calculates an approximation of the marginal likelihood for a model. |
| `CalculateMaxError(Vector<>,Vector<>)` | Calculates the maximum absolute error between actual and predicted values. |
| `CalculateMean(IEnumerable<>)` | Calculates the arithmetic mean (average) of a collection of values. |
| `CalculateMeanAbsoluteDeviation(Vector<>,)` | Calculates the Mean Absolute Deviation (MAD) of a vector of values from a given median. |
| `CalculateMeanAbsoluteError(Vector<>,List<Int32>,List<Int32>)` | Calculates the Mean Absolute Error (MAE) for a split in decision tree algorithms. |
| `CalculateMeanAbsoluteError(Vector<>,Vector<>)` | Calculates the Mean Absolute Error (MAE) between actual and predicted values. |
| `CalculateMeanAbsolutePercentageError(Vector<>,Vector<>)` | Calculates the Mean Absolute Percentage Error (MAPE) between actual and predicted values. |
| `CalculateMeanAndStandardDeviation(Vector<>)` | Calculates the mean (average) and standard deviation of a set of values. |
| `CalculateMeanAveragePrecision(Vector<>,Vector<>,Int32)` | Calculates the Mean Average Precision (MAP) at k for a ranking task. |
| `CalculateMeanBiasError(Vector<>,Vector<>)` | Calculates the mean bias error between actual and predicted values. |
| `CalculateMeanPredictionError(Vector<>,Vector<>)` | Calculates the mean absolute prediction error between actual and predicted values. |
| `CalculateMeanReciprocalRank(Vector<>,Vector<>)` | Calculates the Mean Reciprocal Rank (MRR) for a ranking task. |
| `CalculateMeanSquaredError(IEnumerable<>,)` | Calculates the Mean Squared Error (MSE) between a set of values and their mean. |
| `CalculateMeanSquaredError(IEnumerable<>,IEnumerable<>)` | Calculates the Mean Squared Error (MSE) between actual and predicted values. |
| `CalculateMeanSquaredError(Vector<>,List<Int32>,List<Int32>)` | Calculates the Mean Squared Error (MSE) for a potential split in a decision tree. |
| `CalculateMeanSquaredLogError(Vector<>,Vector<>)` | Calculates the Mean Squared Logarithmic Error (MSLE) between actual and predicted values. |
| `CalculateMedian(IEnumerable<>)` | Calculates the median value from a collection of numeric values. |
| `CalculateMedianAbsoluteDeviation(Vector<>)` | Calculates the median absolute deviation (MAD) of a set of values. |
| `CalculateMedianAbsoluteError(Vector<>,Vector<>)` | Calculates the median absolute error between actual and predicted values. |
| `CalculateMedianPredictionError(Vector<>,Vector<>)` | Calculates the median absolute prediction error between actual and predicted values. |
| `CalculateMinAverageInterClusterDistance(Matrix<>,Vector<>,Int32)` | Calculates the minimum average distance from a point to points in any other cluster. |
| `CalculateMutualInformation(Vector<>,Vector<>)` | Calculates the mutual information between two discrete random variables. |
| `CalculateNDCG(Vector<>,Vector<>,Int32)` | Calculates the Normalized Discounted Cumulative Gain (NDCG) at k for a ranking task. |
| `CalculateNormalCDF(,,)` | Calculates the Cumulative Distribution Function (CDF) for a normal distribution. |
| `CalculateNormalPDF(,,)` | Calculates the probability density function (PDF) for a normal (Gaussian) distribution. |
| `CalculateNormalizedMutualInformation(Vector<>,Vector<>)` | Calculates the normalized mutual information between two variables. |
| `CalculateObservedTestStatistic(Vector<>,Vector<>,TestStatisticType)` | Calculates a test statistic comparing actual and predicted values. |
| `CalculatePValue(Vector<>,Vector<>,TestStatisticType)` | Calculates the p-value for a statistical test comparing two groups. |
| `CalculatePValueFromFDistribution(,Int32,Int32)` | Calculates the p-value from an F-statistic using the F-distribution. |
| `CalculatePValueFromTDistribution(,Int32)` | Calculates the p-value from a t-statistic and degrees of freedom using the t-distribution. |
| `CalculatePValueFromZScore()` | Calculates the p-value from a given z-score in a normal distribution. |
| `CalculatePartialAutoCorrelationFunction(Vector<>,Int32)` | Calculates the partial autocorrelation function (PACF) for a time series up to a specified maximum lag. |
| `CalculatePeakDifference(Vector<>,Vector<>,Vector<>,Vector<>)` | Calculates the difference between peak values in two distributions. |
| `CalculatePearsonCorrelation(Vector<>,Vector<>)` | Calculates the Pearson correlation coefficient between two sets of values. |
| `CalculatePearsonCorrelationCoefficient(Vector<>,Vector<>)` | Calculates the Pearson correlation coefficient between two vectors. |
| `CalculatePercentileInterval(Vector<>,)` | Calculates a percentile-based confidence interval from predicted values. |
| `CalculatePopulationStandardError(Vector<>,Vector<>)` | Calculates the population standard error of the estimate. |
| `CalculatePosteriorPredictiveCheck(ModelStats<,,>)` | Calculates a posterior predictive p-value for model checking. |
| `CalculatePosteriorPredictiveSamples(Vector<>,Vector<>,Int32,Int32)` | Generates samples from the posterior predictive distribution for a model. |
| `CalculatePrecisionRecallAUC(Vector<>,Vector<>)` | Calculates the area under the precision-recall curve (PR AUC). |
| `CalculatePrecisionRecallF1(Vector<>,Vector<>,PredictionType,)` | Calculates precision, recall, and F1 score for a set of predictions. |
| `CalculatePredictionIntervalCoverage(Vector<>,Vector<>,,)` | Calculates the proportion of actual values that fall within a specified prediction interval. |
| `CalculatePredictionIntervals(Vector<>,Vector<>,)` | Calculates prediction intervals for future observations based on a model's predictions. |
| `CalculateQuantile([],)` | Calculates a specific quantile from sorted data. |
| `CalculateQuantileIntervals(Vector<>,Vector<>,[])` | Calculates confidence intervals around specified quantiles of the predicted values. |
| `CalculateQuantiles(Vector<>)` | Calculates the first and third quartiles (25th and 75th percentiles) of a data set. |
| `CalculateR2(Vector<>,Vector<>)` | Calculates the coefficient of determination (R²) between actual and predicted values. |
| `CalculateROCAUC(Vector<>,Vector<>)` | Calculates the Area Under the Receiver Operating Characteristic Curve (ROC AUC). |
| `CalculateROCCurve(Vector<>,Vector<>)` | Calculates the Receiver Operating Characteristic (ROC) curve for a set of predictions. |
| `CalculateRanks(IEnumerable<>)` | Calculates the ranks of values in a collection, handling ties appropriately. |
| `CalculateReferenceModelMarginalLikelihood(Vector<>)` | Calculates the marginal likelihood for a reference model (intercept-only model). |
| `CalculateResidualSumOfSquares(Vector<>,Vector<>)` | Calculates the residual sum of squares (SSE) between actual and predicted values. |
| `CalculateResiduals(Vector<>,Vector<>)` | Calculates the residuals (errors) between actual and predicted values. |
| `CalculateRootMeanSquaredError(Vector<>,Vector<>)` | Calculates the Root Mean Squared Error (RMSE) between actual and predicted values. |
| `CalculateSampleStandardError(Vector<>,Vector<>,Int32)` | Calculates the sample standard error of the estimate, adjusted for the number of model parameters. |
| `CalculateSilhouetteScore(Matrix<>,Vector<>)` | Calculates the silhouette score for a clustering result. |
| `CalculateSimultaneousPredictionInterval(Vector<>,Vector<>,)` | Calculates a simultaneous prediction interval for multiple future observations. |
| `CalculateSkewnessAndKurtosis(Vector<>,,,Int32)` | Calculates the skewness and kurtosis of a sample. |
| `CalculateSpearmanRankCorrelationCoefficient(Vector<>,Vector<>)` | Calculates the Spearman rank correlation coefficient between two vectors. |
| `CalculateSplitScore(Vector<>,List<Int32>,List<Int32>,SplitCriterion)` | Calculates a score for a data split based on the specified criterion. |
| `CalculateStandardDeviation(IEnumerable<>)` | Calculates the standard deviation of a vector. |
| `CalculateStudentPDF(,,,Int32)` | Calculates the probability density function (PDF) value for a Student's t-distribution. |
| `CalculateSymmetricMeanAbsolutePercentageError(Vector<>,Vector<>)` | Calculates the Symmetric Mean Absolute Percentage Error (SMAPE) between actual and predicted values. |
| `CalculateTValue(Int32,)` | Calculates the t-value for a given degrees of freedom and confidence level. |
| `CalculateTheilUStatistic(Vector<>,Vector<>)` | Calculates Theil's U statistic, a measure of forecast accuracy. |
| `CalculateToleranceInterval(Vector<>,Vector<>,)` | Calculates a tolerance interval for a set of predicted values. |
| `CalculateTotalSumOfSquares(Vector<>)` | Calculates the total sum of squares (SST) for a set of values. |
| `CalculateVIF(Matrix<>,ModelStatsOptions)` | Calculates the Variance Inflation Factor (VIF) for each feature based on a correlation matrix. |
| `CalculateVariance(IEnumerable<>)` | Calculates the variance of a collection of values. |
| `CalculateVariance(Vector<>,)` | Calculates the variance of a vector of values from a given mean. |
| `CalculateVarianceReduction(Vector<>,List<Int32>,List<Int32>)` | Calculates the variance reduction achieved by splitting data into left and right groups. |
| `CalculateVariationOfInformation(Vector<>,Vector<>)` | Calculates the variation of information (also known as shared information distance) between two variables. |
| `CalculateWAIC(ModelStats<,,>)` | Calculates the Widely Applicable Information Criterion (WAIC) for Bayesian model comparison. |
| `CalculateWeibullConfidenceIntervals(Vector<>,)` | Calculates confidence intervals for Weibull distribution parameters using bootstrap resampling. |
| `CalculateWeibullCredibleIntervals(Vector<>,,)` | Calculates the credible intervals for a Weibull distribution. |
| `CalculateWeibullPDF(,,)` | Calculates the probability density function (PDF) value for a Weibull distribution. |
| `CalculateWithinClusterVariance(Matrix<>,Vector<>,Dictionary<String,Vector<>>)` | Calculates the within-cluster variance for a clustering result. |
| `ChiSquareCDF(,Int32)` | Calculates the cumulative distribution function (CDF) of the chi-square distribution. |
| `ChiSquarePDF(,Int32)` | Calculates the probability density function (PDF) of the chi-square distribution. |
| `ChiSquarePValue(,Int32)` | Computes the upper-tail p-value of the chi-square distribution: `p = P(χ² > chiSquare \| H0)`. |
| `ChiSquareTest(Vector<>,Vector<>,)` | Performs a Chi-Square test to determine if there is a significant association between two categorical variables. |
| `CosineSimilarity(Vector<>,Vector<>)` | Calculates the cosine similarity between two vectors. |
| `DetermineBestFitDistribution(Vector<>)` | Analyzes a dataset and determines which statistical distribution best fits the data. |
| `Digamma()` | Calculates the digamma function, which is the logarithmic derivative of the gamma function. |
| `Erf()` | Calculates the error function (erf) for a given value. |
| `Erf(Double)` | Error function approximation (Abramowitz and Stegun). |
| `EstimateWeibullParameters(Vector<>)` | Estimates the shape and scale parameters of a Weibull distribution from sample data. |
| `EuclideanDistance(Vector<>,Vector<>)` | Calculates the Euclidean distance between two vectors. |
| `FDistributionPValue(,Int32,Int32)` | Computes the upper-tail (survival-function) p-value of the F-distribution: `p = P(F > fStatistic \| H0)`. |
| `FTest(Vector<>,Vector<>,)` | Performs an F-test to compare the variances of two samples. |
| `FindPeakValue(Vector<>,Vector<>)` | Finds the x-coordinate of the peak value in a distribution. |
| `FitExponentialDistribution(Vector<>)` | Fits an exponential distribution to the provided sample data. |
| `FitLaplaceDistribution(Vector<>)` | Fits a Laplace distribution to the provided data values. |
| `FitLogNormalDistribution(Vector<>)` | Fits a Log-Normal distribution to the provided data values. |
| `FitNormalDistribution(Vector<>)` | Fits a normal distribution to the provided sample data. |
| `FitStudentDistribution(Vector<>)` | Fits a Student's t-distribution to the provided sample data. |
| `FitWeibullDistribution(Vector<>)` | Fits a Weibull distribution to the provided sample data. |
| `Gamma()` | Calculates the gamma function for a given value. |
| `GammaFunction()` | Calculates the Gamma function for a given value. |
| `GammaRegularized(,)` | Calculates the regularized gamma function P(a,x). |
| `GammaRegularizedContinuedFraction(,)` | Calculates the regularized gamma function using a continued fraction expansion. |
| `GammaRegularizedSeries(,)` | Calculates the regularized gamma function using a series expansion. |
| `GeneratePosteriorPredictiveSamples(Matrix<>,Vector<>,Int32)` | Generates samples from the posterior predictive distribution. |
| `GenerateThresholds(Vector<>)` | Generates a set of unique threshold values from predicted values. |
| `HammingDistance(Vector<>,Vector<>)` | Calculates the Hamming distance between two vectors. |
| `IncompleteGamma(,)` | Calculates the incomplete gamma function, which is used in various statistical distributions. |
| `InverseChiSquareCDF(,Int32)` | Calculates the inverse of the Chi-Square cumulative distribution function. |
| `JaccardSimilarity(Vector<>,Vector<>)` | Calculates the Jaccard similarity coefficient between two vectors. |
| `LogBeta(,)` | Calculates the logarithm of the Beta function for parameters a and b. |
| `LogGamma()` | Calculates the logarithm of the Gamma function for parameter x using the Lanczos approximation. |
| `MahalanobisDistance(Vector<>,Vector<>,Matrix<>)` | Calculates the Mahalanobis distance between two vectors, accounting for correlations in the data. |
| `ManhattanDistance(Vector<>,Vector<>)` | Calculates the Manhattan distance (L1 norm) between two vectors. |
| `MannWhitneyUTest(Vector<>,Vector<>,)` | Performs a Mann-Whitney U test to compare distributions between two groups. |
| `MatrixInfinityNorm(Matrix<>)` | Calculates the infinity norm (maximum absolute row sum) of a matrix. |
| `MatrixL1Norm(Matrix<>)` | Calculates the L1 norm (maximum absolute column sum) of a matrix. |
| `NormalCDF(Double)` | Standard normal cumulative distribution function. |
| `NormalPDF(Double)` | Standard normal probability density function. |
| `PermutationTest(Vector<>,Vector<>,)` | Performs a permutation test to determine if there is a significant difference between two groups. |
| `PowerIteration(Matrix<>,Int32,)` | Implements the power iteration method to find the dominant eigenvalue of a matrix. |
| `RegularizedIncompleteBetaFunction(,,)` | Calculates the regularized incomplete Beta function for parameters x, a, and b. |
| `Shuffle(List<>)` | Randomly reorders the elements in a list using the Fisher-Yates shuffle algorithm. |
| `TTest(Vector<>,Vector<>,)` | Performs a Student's t-test to compare means between two groups. |

## Fields

| Field | Summary |
|:-----|:--------|
| `_numOps` | Operations for performing numeric calculations with type T. |

