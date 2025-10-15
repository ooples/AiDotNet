using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace AiDotNet.Enums;

/// <summary>
/// Provides statistical error metrics for model evaluation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations.</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This class calculates various error metrics that help you understand
/// how well your model is performing. It automatically selects appropriate metrics based on
/// your model type or neural network task, and provides convenient access to the results.
/// </para>
/// </remarks>
[Serializable]
public class ErrorStats<T> : StatisticsBase<T>
{
    /// <summary>
    /// Number of features or parameters in the model.
    /// </summary>
    private readonly int _numberOfParameters;

    /// <summary>
    /// The neural network task type, if applicable.
    /// </summary>
    private readonly NeuralNetworkTaskType? _neuralNetworkTaskType;

    /// <summary>
    /// List of individual prediction errors (residuals).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// ErrorList contains the difference between each predicted value and the corresponding actual value.
    /// This lets you examine individual errors, create visualizations like histograms,
    /// or perform additional analyses beyond summary statistics.
    /// </remarks>
    public List<T> ErrorList { get; private set; } = new List<T>();

    #region Property Accessors for Backward Compatibility

    /// <summary>
    /// Mean Absolute Error - The average absolute difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MAE measures the average size of errors without considering their direction (positive or negative).
    /// Lower values indicate better accuracy. If MAE = 5, your predictions are off by 5 units on average.
    /// </remarks>
    public T MAE => GetMetric(MetricType.MAE);

    /// <summary>
    /// Mean Squared Error - The average of squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MSE squares the errors before averaging them, which penalizes large errors more heavily than small ones.
    /// Lower values indicate better accuracy. Because of squaring, the value is not in the same units as your data.
    /// </remarks>
    public T MSE => GetMetric(MetricType.MSE);

    /// <summary>
    /// Root Mean Squared Error - The square root of the Mean Squared Error.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// RMSE converts MSE back to the original units of your data by taking the square root.
    /// It's often preferred over MSE for interpretation because it's in the same units as your data.
    /// Like MAE, lower values indicate better accuracy.
    /// </remarks>
    public T RMSE => GetMetric(MetricType.RMSE);

    /// <summary>
    /// Mean Absolute Percentage Error - The average percentage difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MAPE expresses error as a percentage, which helps you understand the relative size of errors.
    /// MAPE = 10 means that, on average, your predictions are off by 10% from the actual values.
    /// Note: MAPE can be problematic when actual values are close to zero.
    /// </remarks>
    public T MAPE => GetMetric(MetricType.MAPE);

    /// <summary>
    /// Mean Bias Error - The average of prediction errors (predicted - actual).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MeanBiasError helps determine if your model tends to overestimate (positive value) or
    /// underestimate (negative value). Ideally, it should be close to zero, indicating no systematic bias.
    /// Unlike MAE, this doesn't take the absolute value, so positive and negative errors can cancel out.
    /// </remarks>
    public T MeanBiasError => GetMetric(MetricType.MeanBiasError);

    /// <summary>
    /// Median Absolute Error - The middle value of all absolute differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Unlike MAE which uses the average, MedianAbsoluteError uses the middle value of all absolute errors.
    /// This makes it less sensitive to outliers (extreme errors) than MAE.
    /// For example, if you have errors of [1, 2, 100], the median is 2, while the mean would be 34.3.
    /// </remarks>
    public T MedianAbsoluteError => GetMetric(MetricType.MedianAbsoluteError);

    /// <summary>
    /// Maximum Error - The largest absolute difference between any predicted and actual value.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MaxError tells you the worst-case error in your predictions.
    /// It can help you understand the potential maximum impact of prediction errors.
    /// </remarks>
    public T MaxError => GetMetric(MetricType.MaxError);

    /// <summary>
    /// Theil's U Statistic - A measure of forecast accuracy relative to a naive forecasting method.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// TheilUStatistic compares your model's accuracy to a simple "no-change" prediction.
    /// Values less than 1 mean your model is better than the naive approach.
    /// Values equal to 1 mean your model performs the same as the naive approach.
    /// Values greater than 1 mean your model performs worse than the naive approach.
    /// This is especially useful for time series forecasting evaluation.
    /// </remarks>
    public T TheilUStatistic => GetMetric(MetricType.TheilUStatistic);

    /// <summary>
    /// Durbin-Watson Statistic - Detects autocorrelation in prediction errors.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// DurbinWatsonStatistic helps identify if there are patterns in your prediction errors over time.
    /// Values range from 0 to 4:
    /// - Values near 2 suggest no autocorrelation (good)
    /// - Values toward 0 suggest positive autocorrelation (errors tend to be followed by similar errors)
    /// - Values toward 4 suggest negative autocorrelation (errors tend to be followed by opposite errors)
    /// 
    /// Autocorrelation in errors suggests your model might be missing important patterns in the data.
    /// </remarks>
    public T DurbinWatsonStatistic => GetMetric(MetricType.DurbinWatsonStatistic);

    /// <summary>
    /// Sample Standard Error - An estimate of the standard deviation of prediction errors, adjusted for model complexity.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// SampleStandardError estimates how much prediction errors typically vary, taking into account
    /// how many parameters (features) your model uses. It's useful for constructing confidence intervals
    /// around predictions and is adjusted downward based on the number of parameters in your model.
    /// </remarks>
    public T SampleStandardError => GetMetric(MetricType.SampleStandardError);

    /// <summary>
    /// Population Standard Error - The standard deviation of prediction errors without adjustment for model complexity.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// PopulationStandardError measures how much prediction errors typically vary, but unlike 
    /// SampleStandardError, it doesn't adjust for model complexity. It gives you an idea of the
    /// typical size of the errors your model makes.
    /// </remarks>
    public T PopulationStandardError => GetMetric(MetricType.PopulationStandardError);

    /// <summary>
    /// Akaike Information Criterion - A measure that balances model accuracy and complexity.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AIC helps you compare different models by considering both how well they fit the data
    /// and how complex they are. Lower values are better.
    /// 
    /// Think of it like buying a car: You want good performance (accuracy) but don't want to 
    /// pay too much (complexity). AIC helps you find the best balance.
    /// </remarks>
    public T AIC => GetMetric(MetricType.AIC);

    /// <summary>
    /// Bayesian Information Criterion - Similar to AIC but penalizes model complexity more strongly.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// BIC is similar to AIC but tends to prefer simpler models. It helps you compare different
    /// models while avoiding overfitting (when a model memorizes training data instead of learning
    /// general patterns). Lower values are better.
    /// 
    /// BIC is more cautious about adding complexity than AIC, like a budget-conscious car buyer.
    /// </remarks>
    public T BIC => GetMetric(MetricType.BIC);

    /// <summary>
    /// Alternative Akaike Information Criterion - A variant of AIC with a different penalty term.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AICAlt is another version of AIC that uses a slightly different approach to penalize
    /// model complexity. It's particularly useful when sample sizes are small.
    /// Like AIC and BIC, lower values are better.
    /// </remarks>
    public T AICAlt => GetMetric(MetricType.AICAlt);

    /// <summary>
    /// Residual Sum of Squares - The sum of squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// RSS is the total squared error of your model. It's the basis for many other metrics like MSE
    /// (which is just RSS divided by the number of observations).
    /// Lower values indicate a better fit. It's used in calculating metrics like AIC and BIC.
    /// </remarks>
    public T RSS => GetMetric(MetricType.RSS);

    /// <summary>
    /// Area Under the Precision-Recall Curve - Measures classification accuracy focusing on positive cases.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AUCPR is especially useful for imbalanced classification problems (where one class is rare).
    /// It ranges from 0 to 1, with higher values indicating better performance.
    /// 
    /// Precision measures how many of your positive predictions were correct.
    /// Recall measures what fraction of actual positives your model identified.
    /// AUCPR considers how these trade off across different threshold settings.
    /// </remarks>
    public T AUCPR => GetMetric(MetricType.AUCPR);

    /// <summary>
    /// Area Under the Receiver Operating Characteristic Curve - Measures classification accuracy across thresholds.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AUCROC is a common metric for classification models. It ranges from 0 to 1:
    /// - 0.5 means the model is no better than random guessing
    /// - 1.0 means perfect classification
    /// - Values below 0.5 suggest the model is worse than random
    /// 
    /// It measures how well your model can distinguish between classes across different threshold settings.
    /// </remarks>
    public T AUCROC => GetMetric(MetricType.AUCROC);

    /// <summary>
    /// Symmetric Mean Absolute Percentage Error - A variant of MAPE that handles zero or near-zero values better.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// SMAPE is similar to MAPE but uses a different formula that handles cases where actual values are zero
    /// or very small. It's bounded between 0% and 200%, with lower values indicating better performance.
    /// 
    /// SMAPE treats positive and negative errors more symmetrically than MAPE,
    /// which can be important in some forecasting applications.
    /// </remarks>
    public T SMAPE => GetMetric(MetricType.SMAPE);

    /// <summary>
    /// Mean Squared Logarithmic Error - Penalizes underestimates more than overestimates.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MeanSquaredLogError is useful when you care more about relative errors than absolute ones.
    /// It's calculated by applying logarithms to actual and predicted values before computing MSE.
    ///
    /// MSLE penalizes underestimation (predicting too low) more heavily than overestimation.
    /// This is useful in scenarios where underestimating would be more problematic, like inventory forecasting.
    /// </remarks>
    public T MeanSquaredLogError => GetMetric(MetricType.MeanSquaredLogError);

    // Backward compatibility aliases for common property names
    public T Accuracy => GetMetric(MetricType.Accuracy);
    public T F1Score => GetMetric(MetricType.F1Score);
    public T Precision => GetMetric(MetricType.Precision);
    public T Recall => GetMetric(MetricType.Recall);
    public T AUC => GetMetric(MetricType.AUCROC);
    public T MeanSquaredError => MSE;
    public T RootMeanSquaredError => RMSE;
    public T MeanAbsoluteError => MAE;

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new ErrorStats instance and calculates appropriate error metrics based on the model type.
    /// </summary>
    /// <param name="inputs">The inputs containing actual and predicted values.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="neuralNetworkTaskType">Optional. The neural network task type if applicable.</param>
    /// <param name="progress">Optional progress reporting.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    internal ErrorStats(ErrorStatsInputs<T> inputs, ModelType modelType,
                       NeuralNetworkTaskType? neuralNetworkTaskType = null,
                       IProgress<double>? progress = null,
                       CancellationToken cancellationToken = default) : base(modelType)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        _neuralNetworkTaskType = neuralNetworkTaskType;

        if (modelType != ModelType.None)
        {
            DetermineValidMetrics();
        }

        _numberOfParameters = inputs.FeatureCount;

        // Calculate all valid metrics
        if (inputs.Actual != null && inputs.Predicted != null &&
            !inputs.Actual.IsEmpty && !inputs.Predicted.IsEmpty)
        {
            CalculateValidMetrics(inputs.Actual, inputs.Predicted, progress, cancellationToken);
        }
    }

    /// <summary>
    /// Creates an empty ErrorStats instance with appropriate metrics set to zero based on model type.
    /// </summary>
    /// <param name="modelType">The type of model.</param>
    /// <param name="neuralNetworkTaskType">Optional. The neural network task type if applicable.</param>
    /// <returns>An ErrorStats instance with appropriate metrics initialized to zero.</returns>
    /// <remarks>
    /// For Beginners:
    /// This static method creates an ErrorStats object where all metrics that are appropriate
    /// for the specified model type are set to zero. It's useful when you need a placeholder
    /// or default instance, or when you want to compare against a baseline of "no errors."
    /// </remarks>
    public static ErrorStats<T> Empty(ModelType modelType = ModelType.None, NeuralNetworkTaskType? neuralNetworkTaskType = null)
    {
        // Create properly initialized empty inputs
        var emptyInputs = new ErrorStatsInputs<T>
        {
            Actual = Vector<T>.Empty(),
            Predicted = Vector<T>.Empty(),
            FeatureCount = 0
        };

        return new ErrorStats<T>(emptyInputs, modelType, neuralNetworkTaskType);
    }

    /// <summary>
    /// Creates an ErrorStats instance specifically for neural network evaluation.
    /// </summary>
    /// <param name="inputs">The inputs containing actual and predicted values.</param>
    /// <param name="taskType">The specific neural network task type.</param>
    /// <param name="progress">Optional progress reporting.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>An ErrorStats instance configured for the specified neural network task.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method makes it easy to evaluate a neural network by automatically
    /// selecting the appropriate metrics for the specific task your network is performing (like 
    /// image classification, text generation, etc.). You don't need to know which model type to use - 
    /// just specify the task type and this method handles the rest.
    /// </para>
    /// </remarks>
    public static ErrorStats<T> ForNeuralNetwork(
        ErrorStatsInputs<T> inputs,
        NeuralNetworkTaskType taskType,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        // Map the neural network task type to an appropriate model type
        ModelType modelType = NeuralNetworkMetricHelper.MapTaskTypeToModelType(taskType);

        return new ErrorStats<T>(inputs, modelType, taskType, progress, cancellationToken);
    }

    #endregion

    #region Core Calculation Methods

    /// <summary>
    /// Determines which metrics are valid for this statistics provider.
    /// </summary>
    protected override void DetermineValidMetrics()
    {
        _validMetrics.Clear();

        // Check if this is a neural network model with a specified task type
        if (_neuralNetworkTaskType.HasValue &&
            ModelTypeHelper.GetCategory(ModelType) == ModelCategory.NeuralNetwork)
        {
            // Use the neural network validation cache for task-specific metrics
            var cache = NeuralNetworkMetricValidationCache.Instance;
            var taskMetrics = cache.GetValidMetrics(_neuralNetworkTaskType.Value, IsErrorStatisticMetric);

            foreach (var metric in taskMetrics)
            {
                _validMetrics.Add(metric);
            }
        }
        else
        {
            // Use the regular model validation cache for model-specific metrics
            var cache = MetricValidationCache.Instance;
            var modelMetrics = cache.GetValidMetrics(ModelType, IsErrorStatisticMetric);

            foreach (var metric in modelMetrics)
            {
                _validMetrics.Add(metric);
            }
        }
    }

    /// <summary>
    /// Determines if a metric type is a provider-specific statistic metric.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is an error statistic; otherwise, false.</returns>
    protected virtual bool IsProviderStatisticMetric(MetricType metricType)
    {
        return IsErrorStatisticMetric(metricType);
    }

    /// <summary>
    /// Determines if a metric type is an error statistic metric.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is an error statistic; otherwise, false.</returns>
    public static bool IsErrorStatisticMetric(MetricType metricType)
    {
        // Define which metrics are considered error statistics
        return metricType switch
        {
            // Core error metrics
            MetricType.MAE => true,
            MetricType.MSE => true,
            MetricType.RMSE => true,
            MetricType.MAPE => true,
            MetricType.MeanBiasError => true,
            MetricType.MedianAbsoluteError => true,
            MetricType.MaxError => true,

            // Advanced error metrics
            MetricType.TheilUStatistic => true,
            MetricType.DurbinWatsonStatistic => true,
            MetricType.SampleStandardError => true,
            MetricType.PopulationStandardError => true,

            // Model comparison metrics based on errors
            MetricType.AIC => true,
            MetricType.BIC => true,
            MetricType.AICAlt => true,
            MetricType.RSS => true,

            // Classification error metrics
            MetricType.AUCPR => true,
            MetricType.AUCROC => true,

            // Alternative error metrics
            MetricType.SMAPE => true,
            MetricType.MeanSquaredLogError => true,

            // Neural network specific metrics
            MetricType.CrossEntropyLoss => true,
            MetricType.LogLikelihood => true,
            MetricType.Perplexity => true,
            MetricType.KLDivergence => true,

            // Time series metrics
            MetricType.DynamicTimeWarping => true,

            // For any other metric type
            _ => false,
        };
    }

    /// <summary>
    /// Calculates all metrics that are valid for the current model type.
    /// </summary>
    private void CalculateValidMetrics(Vector<T> actual, Vector<T> predicted,
                                      IProgress<double>? progress = null,
                                      CancellationToken cancellationToken = default)
    {
        try
        {
            // Validate inputs
            if (actual.Length != predicted.Length)
            {
                throw new ArgumentException("Actual and predicted vectors must have the same length.");
            }

            // Calculate residuals for all model types
            ErrorList = new List<T>(StatisticsHelper<T>.CalculateResiduals(actual, predicted));

            // Check for cancellation
            cancellationToken.ThrowIfCancellationRequested();

            // Report initial progress
            progress?.Report(0);

            // Calculate each valid metric
            int total = _validMetrics.Count;
            int completed = 0;

            // Decide whether to use parallel or sequential calculation based on data size
            if (actual.Length > 10000 && total > 5)
            {
                // For large datasets with many metrics, parallel calculation may be more efficient
                var options = new ParallelOptions
                {
                    CancellationToken = cancellationToken,
                    MaxDegreeOfParallelism = Environment.ProcessorCount
                };

                var metrics = new Dictionary<MetricType, T>();
                var syncLock = new object();

                Parallel.ForEach(_validMetrics, options, metricType =>
                {
                    var value = CalculateMetricValue(metricType, actual, predicted);

                    lock (syncLock)
                    {
                        metrics[metricType] = value;
                        completed++;
                        progress?.Report((double)completed / total);
                    }
                });

                // After parallel calculation, update the dictionary
                foreach (var kv in metrics)
                {
                    _metrics[kv.Key] = kv.Value;
                    _calculatedMetrics.Add(kv.Key);
                }
            }
            else
            {
                // For smaller datasets, sequential calculation is simpler and may be fast enough
                foreach (var metricType in _validMetrics)
                {
                    // Check for cancellation
                    cancellationToken.ThrowIfCancellationRequested();

                    var value = CalculateMetricValue(metricType, actual, predicted);
                    _metrics[metricType] = value;
                    _calculatedMetrics.Add(metricType);

                    completed++;
                    progress?.Report((double)completed / total);
                }
            }

            // After calculating primary metrics, calculate dependent ones
            CalculateDependentMetrics();
        }
        catch (OperationCanceledException)
        {
            // Propagate cancellation
            throw;
        }
        catch (Exception ex)
        {
            // Wrap in a custom exception with context information
            throw new ErrorStatsException($"Error calculating metrics for model type {ModelType}", ex);
        }
    }

    /// <summary>
    /// Calculates metrics that depend on other metrics, ensuring proper calculation order.
    /// </summary>
    private void CalculateDependentMetrics()
    {
        int n = 0;
        if (ErrorList != null && ErrorList.Count > 0)
        {
            n = ErrorList.Count;
        }

        // RMSE depends on MSE
        if (_calculatedMetrics.Contains(MetricType.MSE) &&
            _validMetrics.Contains(MetricType.RMSE) &&
            !_calculatedMetrics.Contains(MetricType.RMSE))
        {
            var mse = _metrics[MetricType.MSE];
            _metrics[MetricType.RMSE] = _numOps.Sqrt(mse);
            _calculatedMetrics.Add(MetricType.RMSE);
        }

        // AIC, BIC, and AICAlt depend on RSS and sample size
        if (_calculatedMetrics.Contains(MetricType.RSS) && n > 0)
        {
            var rss = _metrics[MetricType.RSS];

            // AIC calculation
            if (_validMetrics.Contains(MetricType.AIC) && !_calculatedMetrics.Contains(MetricType.AIC))
            {
                _metrics[MetricType.AIC] = StatisticsHelper<T>.CalculateAIC(n, _numberOfParameters, rss);
                _calculatedMetrics.Add(MetricType.AIC);
            }

            // BIC calculation
            if (_validMetrics.Contains(MetricType.BIC) && !_calculatedMetrics.Contains(MetricType.BIC))
            {
                _metrics[MetricType.BIC] = StatisticsHelper<T>.CalculateBIC(n, _numberOfParameters, rss);
                _calculatedMetrics.Add(MetricType.BIC);
            }

            // AICAlt calculation
            if (_validMetrics.Contains(MetricType.AICAlt) && !_calculatedMetrics.Contains(MetricType.AICAlt))
            {
                _metrics[MetricType.AICAlt] = StatisticsHelper<T>.CalculateAICAlternative(n, _numberOfParameters, rss);
                _calculatedMetrics.Add(MetricType.AICAlt);
            }
        }

        // PopulationStandardError could be optimized if MSE is already calculated
        if (_calculatedMetrics.Contains(MetricType.MSE) &&
            _validMetrics.Contains(MetricType.PopulationStandardError) &&
            !_calculatedMetrics.Contains(MetricType.PopulationStandardError) &&
            n > 0)
        {
            var mse = _metrics[MetricType.MSE];
            _metrics[MetricType.PopulationStandardError] = _numOps.Sqrt(mse);
            _calculatedMetrics.Add(MetricType.PopulationStandardError);
        }

        // SampleStandardError could be optimized if PopulationStandardError is already calculated
        if (_calculatedMetrics.Contains(MetricType.PopulationStandardError) &&
            _validMetrics.Contains(MetricType.SampleStandardError) &&
            !_calculatedMetrics.Contains(MetricType.SampleStandardError) &&
            n > _numberOfParameters)
        {
            var popStdErr = _metrics[MetricType.PopulationStandardError];
            var correctionFactor = _numOps.FromDouble((double)(n) / (n - _numberOfParameters));
            correctionFactor = _numOps.Sqrt(correctionFactor);
            _metrics[MetricType.SampleStandardError] = _numOps.Multiply(popStdErr, correctionFactor);
            _calculatedMetrics.Add(MetricType.SampleStandardError);
        }

        // Check if we need to update any other dependent metrics based on our error list
        if (ErrorList != null && ErrorList.Count > 0 && !_calculatedMetrics.Contains(MetricType.MaxError) &&
            _validMetrics.Contains(MetricType.MaxError))
        {
            // Calculate MaxError from the error list if we have it
            var maxError = _numOps.Zero;
            bool first = true;

            foreach (var error in ErrorList)
            {
                var absError = _numOps.Abs(error);
                if (first || _numOps.GreaterThan(absError, maxError))
                {
                    maxError = absError;
                    first = false;
                }
            }

            _metrics[MetricType.MaxError] = maxError;
            _calculatedMetrics.Add(MetricType.MaxError);
        }
    }

    /// <summary>
    /// Calculates the value of a specific metric.
    /// </summary>
    private T CalculateMetricValue(MetricType metricType, Vector<T> actual, Vector<T> predicted)
    {
        try
        {
            switch (metricType)
            {
                case MetricType.MAE:
                    return StatisticsHelper<T>.CalculateMeanAbsoluteError(actual, predicted);

                case MetricType.MSE:
                    return StatisticsHelper<T>.CalculateMeanSquaredError(actual, predicted);

                case MetricType.RMSE:
                    var mse = StatisticsHelper<T>.CalculateMeanSquaredError(actual, predicted);
                    return _numOps.Sqrt(mse);

                case MetricType.MAPE:
                    return StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(actual, predicted);

                case MetricType.MeanBiasError:
                    return StatisticsHelper<T>.CalculateMeanBiasError(actual, predicted);

                case MetricType.MedianAbsoluteError:
                    return StatisticsHelper<T>.CalculateMedianAbsoluteError(actual, predicted);

                case MetricType.MaxError:
                    return StatisticsHelper<T>.CalculateMaxError(actual, predicted);

                case MetricType.TheilUStatistic:
                    return StatisticsHelper<T>.CalculateTheilUStatistic(actual, predicted);

                case MetricType.DurbinWatsonStatistic:
                    return StatisticsHelper<T>.CalculateDurbinWatsonStatistic(actual, predicted);

                case MetricType.SampleStandardError:
                    return StatisticsHelper<T>.CalculateSampleStandardError(actual, predicted, _numberOfParameters);

                case MetricType.PopulationStandardError:
                    return StatisticsHelper<T>.CalculatePopulationStandardError(actual, predicted);

                case MetricType.AIC:
                    var rss = StatisticsHelper<T>.CalculateResidualSumOfSquares(actual, predicted);
                    return StatisticsHelper<T>.CalculateAIC(actual.Length, _numberOfParameters, rss);

                case MetricType.BIC:
                    rss = StatisticsHelper<T>.CalculateResidualSumOfSquares(actual, predicted);
                    return StatisticsHelper<T>.CalculateBIC(actual.Length, _numberOfParameters, rss);

                case MetricType.AICAlt:
                    rss = StatisticsHelper<T>.CalculateResidualSumOfSquares(actual, predicted);
                    return StatisticsHelper<T>.CalculateAICAlternative(actual.Length, _numberOfParameters, rss);

                case MetricType.RSS:
                    return StatisticsHelper<T>.CalculateResidualSumOfSquares(actual, predicted);

                case MetricType.AUCPR:
                    return StatisticsHelper<T>.CalculatePrecisionRecallAUC(actual, predicted);

                case MetricType.AUCROC:
                    return StatisticsHelper<T>.CalculateROCAUC(actual, predicted);

                case MetricType.SMAPE:
                    return StatisticsHelper<T>.CalculateSymmetricMeanAbsolutePercentageError(actual, predicted);

                case MetricType.MeanSquaredLogError:
                    return StatisticsHelper<T>.CalculateMeanSquaredLogError(actual, predicted);

                case MetricType.CrossEntropyLoss:
                    return StatisticsHelper<T>.CalculateCrossEntropyLoss(actual, predicted);

                case MetricType.LogLikelihood:
                    return StatisticsHelper<T>.CalculateLogLikelihood(actual, predicted);

                case MetricType.Perplexity:
                    return StatisticsHelper<T>.CalculatePerplexity(actual, predicted);

                case MetricType.KLDivergence:
                    return StatisticsHelper<T>.CalculateKLDivergence(actual, predicted);

                case MetricType.DynamicTimeWarping:
                    return StatisticsHelper<T>.CalculateDynamicTimeWarping(actual, predicted);

                default:
                    throw new NotImplementedException($"Calculation for metric {metricType} is not implemented.");
            }
        }
        catch (Exception ex)
        {
            // Wrap in a custom exception with context information
            throw new ErrorStatsException($"Error calculating metric {metricType}", ex);
        }
    }

    #endregion

    #region Public API Methods

    /// <summary>
    /// Tries to get the value of a specific metric.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <param name="value">The value of the requested metric if successful.</param>
    /// <returns>True if the metric was successfully retrieved; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This is a safer version of GetMetric that won't throw an exception if the metric
    /// is invalid or hasn't been calculated. Instead, it returns false and sets the output
    /// value to zero. This pattern is commonly used in production code to avoid exceptions.
    /// </para>
    /// </remarks>
    public bool TryGetMetric(MetricType metricType, out double value)
    {
        if (!IsValidMetric(metricType))
        {
            value = Convert.ToDouble(_numOps.Zero);
            return false;
        }

        if (_metrics.TryGetValue(metricType, out var metricValue))
        {
            value = Convert.ToDouble(metricValue);
            return true;
        }

        value = Convert.ToDouble(_numOps.Zero);
        return false;
    }

    /// <summary>
    /// Gets information about which neural network task type was used to select metrics.
    /// </summary>
    /// <returns>The neural network task type if one was specified, or null if not applicable.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method tells you which neural network task type (like image classification
    /// or text generation) was used to determine the appropriate metrics. This can be helpful for 
    /// understanding why certain metrics were calculated.
    /// </para>
    /// </remarks>
    public NeuralNetworkTaskType? GetNeuralNetworkTaskType()
    {
        return _neuralNetworkTaskType;
    }

    /// <summary>
    /// Gets a descriptive name for this error stats instance based on model type or neural network task.
    /// </summary>
    /// <returns>A user-friendly description of the model or task being evaluated.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method provides a readable name describing what kind of model
    /// or neural network task these error statistics are for. It's useful for labeling reports,
    /// charts, or logs.
    /// </para>
    /// </remarks>
    public string GetDescriptiveName()
    {
        if (_neuralNetworkTaskType.HasValue)
        {
            return NeuralNetworkMetricHelper.GetTaskName(_neuralNetworkTaskType.Value);
        }
        else
        {
            return $"{ModelType} Evaluation";
        }
    }

    #endregion

    #region Serialization

    /// <summary>
    /// Serializes the error stats to JSON.
    /// </summary>
    /// <returns>A JSON string representing the error stats.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method converts the error statistics into a JSON string format, which is easy to
    /// store in files or databases, or to send over networks. It's useful when you need to
    /// save your results or share them with other applications.
    /// </para>
    /// </remarks>
    public string ToJson()
    {
        var result = new Dictionary<string, object>
        {
            ["ModelType"] = ModelType.ToString(),
            ["Metrics"] = System.Linq.Enumerable.ToDictionary(_metrics, kv => kv.Key.ToString(), kv => kv.Value),
            ["ValidMetrics"] = _validMetrics.Select(m => m.ToString()).ToArray(),
            ["CalculatedMetrics"] = _calculatedMetrics.Select(m => m.ToString()).ToArray()
        };

        if (_neuralNetworkTaskType.HasValue)
        {
            result["NeuralNetworkTaskType"] = _neuralNetworkTaskType.Value.ToString();
        }

        return JsonConvert.SerializeObject(result);
    }

    #endregion
}