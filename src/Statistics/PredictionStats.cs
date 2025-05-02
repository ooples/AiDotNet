namespace AiDotNet.Statistics;

/// <summary>
/// Calculates and stores various statistics to evaluate prediction performance and generate prediction intervals.
/// Only calculates metrics appropriate for the specific model type.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
[Serializable]
public class PredictionStats<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// The type of model being evaluated.
    /// </summary>
    public ModelType ModelType { get; private set; }

    /// <summary>
    /// Number of features or parameters in the model.
    /// </summary>
    private readonly int _numberOfParameters;

    /// <summary>
    /// The confidence level used for statistical intervals.
    /// </summary>
    private readonly T _confidenceLevel;

    /// <summary>
    /// The type of prediction (regression, binary classification, etc.)
    /// </summary>
    private readonly PredictionType _predictionType;

    /// <summary>
    /// Dictionary to store calculated metrics.
    /// </summary>
    private readonly Dictionary<MetricType, T> _metrics = [];

    /// <summary>
    /// Set of metrics that have been calculated.
    /// </summary>
    private readonly HashSet<MetricType> _calculatedMetrics = [];

    /// <summary>
    /// Set of metrics valid for this model type.
    /// </summary>
    private readonly HashSet<MetricType> _validMetrics = [];

    /// <summary>
    /// Dictionary to store calculated intervals as tuples.
    /// </summary>
    private readonly Dictionary<IntervalType, (T Lower, T Upper)> _intervals = [];

    /// <summary>
    /// Set of intervals that have been calculated.
    /// </summary>
    private readonly HashSet<IntervalType> _calculatedIntervals = [];

    /// <summary>
    /// Set of intervals valid for this model type.
    /// </summary>
    private readonly HashSet<IntervalType> _validIntervals = [];

    #region Property Accessors

    /// <summary>
    /// Prediction Interval - A range that likely contains future individual observations.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// A prediction interval gives you a range where future individual values are likely to fall.
    /// For example, if your model predicts a value of 100 and the prediction interval is (90, 110),
    /// you can be reasonably confident that the actual value will be between 90 and 110.
    /// 
    /// Unlike confidence intervals (which are about the average), prediction intervals account for
    /// both the uncertainty in the average prediction and the natural variability of individual values.
    /// </remarks>
    public (T Lower, T Upper) PredictionInterval => GetInterval(IntervalType.Prediction);

    /// <summary>
    /// Confidence Interval - A range that likely contains the true mean of the predictions.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// A confidence interval tells you the range where the true average prediction is likely to be.
    /// It helps you understand the precision of your model's average prediction.
    /// 
    /// For example, if your model predicts an average of 50 and the confidence interval is (48, 52),
    /// you can be reasonably confident that the true average is between 48 and 52.
    /// The interval gets narrower with more data, indicating more precise estimates.
    /// </remarks>
    public (T Lower, T Upper) ConfidenceInterval => GetInterval(IntervalType.Confidence);

    /// <summary>
    /// Credible Interval - A Bayesian interval that contains the true value with a certain probability.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// A credible interval is the Bayesian version of a confidence interval. While they look similar,
    /// they have different interpretations:
    /// 
    /// You can say "There's a 95% chance that the true value lies within this credible interval,"
    /// which is a more intuitive interpretation than confidence intervals provide.
    /// 
    /// Credible intervals incorporate prior knowledge about the parameter being estimated,
    /// which can be beneficial when you have domain expertise or previous data.
    /// </remarks>
    public (T Lower, T Upper) CredibleInterval => GetInterval(IntervalType.Credible);

    /// <summary>
    /// Tolerance Interval - A range that contains a specified proportion of the population with a certain confidence.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// A tolerance interval is different from both prediction and confidence intervals.
    /// It gives you a range that contains a specific percentage of all possible values
    /// (not just the average) with a certain level of confidence.
    /// 
    /// For example, a 95/99 tolerance interval means you're 95% confident that the interval
    /// contains 99% of all possible values from the population.
    /// 
    /// These are useful when you need to understand the range of almost all possible values,
    /// such as in quality control or setting specification limits.
    /// </remarks>
    public (T Lower, T Upper) ToleranceInterval => GetInterval(IntervalType.Tolerance);

    /// <summary>
    /// Forecast Interval - A prediction interval specifically for time series forecasting.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Forecast intervals are similar to prediction intervals but are specifically designed
    /// for time series data (data collected over time, like monthly sales or daily temperatures).
    /// 
    /// They account for the unique characteristics of time series data, like increasing uncertainty
    /// the further you forecast into the future and potential seasonal patterns.
    /// 
    /// A wider forecast interval indicates less certainty about future values.
    /// </remarks>
    public (T Lower, T Upper) ForecastInterval => GetInterval(IntervalType.Forecast);

    /// <summary>
    /// Bootstrap Interval - An interval created using resampling techniques.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Bootstrap intervals use a technique called "bootstrapping," where many samples are
    /// randomly drawn (with replacement) from your data to estimate the variability in your predictions.
    /// 
    /// This approach is powerful because it doesn't assume any particular distribution for your data,
    /// making it robust when your data doesn't follow common patterns like the normal distribution.
    /// 
    /// Bootstrap intervals are especially useful when you have limited data or when the theoretical
    /// assumptions for other interval types might not be valid.
    /// </remarks>
    public (T Lower, T Upper) BootstrapInterval => GetInterval(IntervalType.Bootstrap);

    /// <summary>
    /// Simultaneous Prediction Interval - A prediction interval that accounts for multiple predictions.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// When you make multiple predictions at once, there's a higher chance that at least one of them
    /// will fall outside a standard prediction interval just by random chance.
    /// 
    /// Simultaneous prediction intervals account for this by creating wider intervals that are guaranteed
    /// to contain a certain percentage of all predictions made simultaneously.
    /// 
    /// These are important when you need to ensure that all (or most) of your predictions are within
    /// the intervals, not just each one individually.
    /// </remarks>
    public (T Lower, T Upper) SimultaneousPredictionInterval => GetInterval(IntervalType.SimultaneousPrediction);

    /// <summary>
    /// Jackknife Interval - An interval created by systematically leaving out one observation at a time.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// The jackknife method creates an interval by repeatedly recalculating your statistics
    /// while leaving out one data point each time. This helps assess how sensitive your results
    /// are to individual data points.
    /// 
    /// It's particularly useful for detecting outliers or influential points that might be
    /// skewing your results, and for creating intervals when sample sizes are small.
    /// 
    /// Like bootstrap intervals, jackknife intervals don't make strong assumptions about
    /// the distribution of your data.
    /// </remarks>
    public (T Lower, T Upper) JackknifeInterval => GetInterval(IntervalType.Jackknife);

    /// <summary>
    /// Percentile Interval - An interval based directly on the percentiles of the prediction distribution.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// A percentile interval is one of the simplest types of intervals, created by taking
    /// the percentiles directly from your prediction distribution.
    /// 
    /// For example, a 95% percentile interval might use the 2.5th and 97.5th percentiles
    /// of your predictions as the lower and upper bounds.
    /// 
    /// These intervals are intuitive and don't require many statistical assumptions,
    /// making them useful for quick assessments of your prediction range.
    /// </remarks>
    public (T Lower, T Upper) PercentileInterval => GetInterval(IntervalType.Percentile);

    /// <summary>
    /// Collection of prediction intervals at different quantile levels.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// QuantileIntervals provides prediction intervals at different probability levels.
    /// 
    /// For example, it might include a 50% interval (showing where the middle half of values are expected),
    /// a 75% interval, and a 95% interval.
    /// 
    /// Each tuple contains:
    /// - Quantile: The probability level (like 0.5 for 50%)
    /// - Lower: The lower bound of the interval
    /// - Upper: The upper bound of the interval
    /// 
    /// These are useful for understanding the uncertainty at different levels of confidence.
    /// </remarks>
    public List<(T Quantile, T Lower, T Upper)> QuantileIntervals { get; private set; } = [];

    /// <summary>
    /// The proportion of actual values that fall within the prediction interval.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// PredictionIntervalCoverage tells you how well your prediction intervals actually work.
    /// 
    /// For example, if you have a 95% prediction interval, ideally about 95% of actual values
    /// should fall within those intervals. PredictionIntervalCoverage measures the actual percentage.
    /// 
    /// If this value is much lower than expected (like 70% for a 95% interval), your intervals
    /// are too narrow. If it's much higher (like 99% for a 95% interval), your intervals
    /// might be unnecessarily wide.
    /// </remarks>
    public T PredictionIntervalCoverage => GetMetric(MetricType.PredictionIntervalCoverage);

    /// <summary>
    /// The average of all prediction errors (predicted - actual).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MeanPredictionError measures the average difference between predicted and actual values.
    /// 
    /// A value close to zero suggests your model doesn't systematically overestimate or underestimate.
    /// A positive value means your model tends to predict values higher than the actual values.
    /// A negative value means your model tends to predict values lower than the actual values.
    /// 
    /// Unlike error metrics that use absolute values (like MAE), this can tell you about the
    /// direction of the error, but positive and negative errors can cancel each other out.
    /// </remarks>
    public T MeanPredictionError => GetMetric(MetricType.MeanPredictionError);

    /// <summary>
    /// The middle value of all prediction errors (predicted - actual).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Similar to MeanPredictionError, but uses the median (middle value) instead of the mean (average).
    /// 
    /// The advantage of using the median is that it's less affected by outliers or extreme errors.
    /// For example, if most errors are small but one is very large, the median will still show
    /// a value representative of the typical error.
    /// 
    /// Like MeanPredictionError, a value close to zero is ideal.
    /// </remarks>
    public T MedianPredictionError => GetMetric(MetricType.MedianPredictionError);

    /// <summary>
    /// Coefficient of determination - The proportion of variance in the dependent variable explained by the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// R² (R-squared) is perhaps the most common metric for regression models. It ranges from 0 to 1:
    /// - 1 means your model perfectly predicts all values
    /// - 0 means your model does no better than simply predicting the average for every case
    /// - Values in between indicate the percentage of variance your model explains
    /// 
    /// For example, an R² of 0.75 means your model explains 75% of the variability in the target variable.
    /// 
    /// Be careful: a high R² doesn't necessarily mean your model is good - it could be overfitting!
    /// </remarks>
    public T R2 => GetMetric(MetricType.R2);

    /// <summary>
    /// R² adjusted for the number of predictors in the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AdjustedR2 is a modified version of R² that accounts for the number of features in your model.
    /// 
    /// Regular R² always increases when you add more features, even if they don't actually improve predictions.
    /// AdjustedR2 penalizes adding unnecessary features, so it only increases if the new feature
    /// actually improves the model more than would be expected by chance.
    /// 
    /// This makes it more useful when comparing models with different numbers of features.
    /// Like R², values closer to 1 are better.
    /// </remarks>
    public T AdjustedR2 => GetMetric(MetricType.AdjustedR2);

    /// <summary>
    /// The explained variance score - A measure of how well the model accounts for the variance in the data.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// ExplainedVarianceScore is similar to R², but it doesn't penalize the model for systematic bias.
    /// 
    /// It ranges from 0 to 1, with higher values being better:
    /// - 1 means your model explains all the variance in the data (perfect)
    /// - 0 means your model doesn't explain any variance
    /// 
    /// If your model's predictions are all shifted by a constant amount from the actual values,
    /// R² would be lower, but ExplainedVarianceScore would still be high.
    /// </remarks>
    public T ExplainedVarianceScore => GetMetric(MetricType.ExplainedVarianceScore);

    /// <summary>
    /// A list of performance metrics calculated at different training set sizes.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// LearningCurve shows how your model's performance changes as you give it more training data.
    /// 
    /// This is helpful for diagnosing if your model is suffering from high bias (underfitting)
    /// or high variance (overfitting):
    /// - If performance quickly plateaus with small amounts of data, you might have high bias
    /// - If performance continues improving with more data, you might need even more data
    /// - If there's a large gap between training and validation performance, you might have high variance
    /// 
    /// The list contains performance metrics at different training set sizes.
    /// </remarks>
    public List<T> LearningCurve { get; private set; } = [];

    /// <summary>
    /// The proportion of predictions that the model got correct (for classification).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Accuracy is a simple metric for classification problems. It's the percentage of predictions
    /// that match the actual values.
    /// 
    /// For example, if your model correctly classifies 90 out of 100 samples, the accuracy is 0.9 or 90%.
    /// 
    /// While intuitive, accuracy can be misleading for imbalanced classes. For example, if 95% of your
    /// data belongs to class A, a model that always predicts class A would have 95% accuracy
    /// despite being useless for class B.
    /// </remarks>
    public T Accuracy => GetMetric(MetricType.Accuracy);

    /// <summary>
    /// The proportion of positive predictions that were actually correct (for classification).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Precision answers the question: "Of all the items labeled as positive, how many actually were positive?"
    /// 
    /// It ranges from 0 to 1, with 1 being perfect.
    /// 
    /// For example, if your model identifies 100 emails as spam, and 90 of them actually are spam,
    /// the precision is 0.9 or 90%.
    /// 
    /// Precision is important when the cost of false positives is high. In the spam example,
    /// high precision means fewer important emails mistakenly marked as spam.
    /// </remarks>
    public T Precision => GetMetric(MetricType.Precision);

    /// <summary>
    /// The proportion of actual positive cases that were correctly identified (for classification).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Recall answers the question: "Of all the actual positive items, how many did the model identify?"
    /// 
    /// It ranges from 0 to 1, with 1 being perfect.
    /// 
    /// For example, if there are 100 spam emails, and your model identifies 80 of them,
    /// the recall is 0.8 or 80%.
    /// 
    /// Recall is important when the cost of false negatives is high. In a medical context,
    /// high recall means catching most cases of a disease, even if it means some false alarms.
    /// </remarks>
    public T Recall => GetMetric(MetricType.Recall);

    /// <summary>
    /// The harmonic mean of precision and recall (for classification).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// F1Score balances precision and recall in a single metric, which is helpful because
    /// there's often a trade-off between them.
    /// 
    /// It ranges from 0 to 1, with 1 being perfect.
    /// 
    /// F1Score is particularly useful when:
    /// - You need a single metric to compare models
    /// - Classes are imbalanced (one class is much more common than others)
    /// - You care equally about false positives and false negatives
    /// 
    /// It's calculated as 2 * (precision * recall) / (precision + recall).
    /// </remarks>
    public T F1Score => GetMetric(MetricType.F1Score);

    /// <summary>
    /// A measure of linear correlation between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// PearsonCorrelation measures the strength and direction of the linear relationship
    /// between actual and predicted values.
    /// 
    /// It ranges from -1 to 1:
    /// - 1 means perfect positive correlation (as actual increases, predicted increases proportionally)
    /// - 0 means no correlation
    /// - -1 means perfect negative correlation (as actual increases, predicted decreases proportionally)
    /// 
    /// For prediction models, you typically want values close to 1, indicating that your
    /// predictions track well with actual values.
    /// </remarks>
    public T PearsonCorrelation => GetMetric(MetricType.PearsonCorrelation);

    /// <summary>
    /// A measure of monotonic correlation between the ranks of actual and predicted values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// SpearmanCorrelation is similar to PearsonCorrelation but works with ranks instead of raw values.
    /// 
    /// It measures whether predicted values tend to increase when actual values increase,
    /// regardless of whether the relationship is linear.
    /// 
    /// Like Pearson's correlation, it ranges from -1 to 1:
    /// - 1 means perfect positive rank correlation
    /// - 0 means no rank correlation
    /// - -1 means perfect negative rank correlation
    /// 
    /// This is useful when you care about the order of predictions more than their exact values.
    /// </remarks>
    public T SpearmanCorrelation => GetMetric(MetricType.SpearmanCorrelation);

    /// <summary>
    /// A measure of concordance between actual and predicted values based on paired rankings.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// KendallTau is another rank correlation metric, similar to SpearmanCorrelation.
    /// 
    /// It measures the proportion of pairs that are ranked in the same order in both actual
    /// and predicted values.
    /// 
    /// Like other correlation metrics, it ranges from -1 to 1:
    /// - 1 means all pairs have the same ordering in both sets
    /// - 0 means the orderings are random relative to each other
    /// - -1 means the orderings are exactly opposite
    /// 
    /// KendallTau is often more robust to outliers than SpearmanCorrelation.
    /// </remarks>
    public T KendallTau => GetMetric(MetricType.KendallTau);

    /// <summary>
    /// A measure of similarity between two temporal sequences.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// DynamicTimeWarping measures how similar two sequences are, even if they're not perfectly aligned in time.
    /// 
    /// Unlike simple point-by-point comparisons, it can handle sequences that are stretched,
    /// compressed, or shifted in time. This is particularly useful for comparing time series
    /// where patterns might occur at different rates.
    /// 
    /// Lower values indicate more similar sequences.
    /// 
    /// This is especially useful for time series data like speech recognition, gesture recognition,
    /// or any data where timing may vary.
    /// </remarks>
    public T DynamicTimeWarping => GetMetric(MetricType.DynamicTimeWarping);

    /// <summary>
    /// Information about the statistical distribution that best fits the prediction data.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// BestDistributionFit helps you understand the shape of your prediction distribution.
    /// 
    /// Knowing the distribution can help with:
    /// - Creating better intervals (by using the right distribution assumptions)
    /// - Understanding the types of predictions your model makes (are they normally distributed? skewed?)
    /// - Identifying potential issues with your predictions
    /// 
    /// This object contains information about which distribution type best fits your data
    /// and parameters describing that distribution.
    /// </remarks>
    public DistributionFitResult<T> BestDistributionFit { get; private set; } = new();

    #endregion

    #region Constructors

    /// <summary>
    /// Creates a new PredictionStats instance and calculates appropriate prediction metrics based on the model type.
    /// </summary>
    /// <param name="inputs">The inputs containing actual and predicted values.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor takes your actual values (ground truth), predicted values,
    /// and the type of model you're evaluating. It then calculates only the prediction metrics
    /// that are appropriate for that type of model.
    /// </remarks>
    internal PredictionStats(PredictionStatsInputs<T> inputs, ModelType modelType)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        _numOps = MathHelper.GetNumericOperations<T>();
        ModelType = modelType;
        _numberOfParameters = inputs.NumberOfParameters;
        _confidenceLevel = _numOps.FromDouble(inputs.ConfidenceLevel);
        _predictionType = inputs.PredictionType;

        // Determine which metrics and intervals are valid for this model type
        DetermineValidMetricsAndIntervals();

        // Calculate valid metrics and intervals
        if (inputs.Actual != null && inputs.Predicted != null &&
            !inputs.Actual.IsEmpty && !inputs.Predicted.IsEmpty)
        {
            CalculateValidMetricsAndIntervals(inputs.Actual, inputs.Predicted, inputs.LearningCurveSteps);
        }
    }

    /// <summary>
    /// Creates a new PredictionStats instance and calculates appropriate prediction metrics based on the model type.
    /// </summary>
    /// <param name="inputs">The inputs containing actual and predicted values.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="progress">Optional progress reporting.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    internal PredictionStats(PredictionStatsInputs<T> inputs, ModelType modelType,
                            IProgress<double>? progress = null,
                            CancellationToken cancellationToken = default)
    {
        if (inputs == null)
            throw new ArgumentNullException(nameof(inputs));

        _numOps = MathHelper.GetNumericOperations<T>();
        ModelType = modelType;
        _numberOfParameters = inputs.NumberOfParameters;
        _confidenceLevel = _numOps.FromDouble(inputs.ConfidenceLevel);
        _predictionType = inputs.PredictionType;

        // Determine which metrics and intervals are valid for this model type
        DetermineValidMetricsAndIntervals();

        // Calculate valid metrics and intervals
        if (inputs.Actual != null && inputs.Predicted != null &&
            !inputs.Actual.IsEmpty && !inputs.Predicted.IsEmpty)
        {
            CalculateValidMetricsAndIntervals(inputs.Actual, inputs.Predicted, inputs.LearningCurveSteps,
                                             progress, cancellationToken);
        }
    }

    /// <summary>
    /// Creates an empty PredictionStats instance with appropriate metrics set to zero based on model type.
    /// </summary>
    /// <param name="modelType">The type of model.</param>
    /// <returns>A PredictionStats instance with appropriate metrics initialized to zero.</returns>
    /// <remarks>
    /// For Beginners:
    /// This static method creates a PredictionStats object where all metrics that are appropriate
    /// for the specified model type are set to zero. It's useful when you need a placeholder
    /// or default instance, or when you want to compare against a baseline of "perfect predictions."
    /// </remarks>
    public static PredictionStats<T> Empty(ModelType modelType)
    {
        // Create properly initialized empty inputs
        var emptyInputs = new PredictionStatsInputs<T>
        {
            Actual = Vector<T>.Empty(),
            Predicted = Vector<T>.Empty(),
            ConfidenceLevel = 0.95, // Default confidence level
            NumberOfParameters = 0,
            LearningCurveSteps = 0,
            PredictionType = PredictionType.Regression // Default prediction type
        };

        return new PredictionStats<T>(emptyInputs, modelType);
    }

    #endregion

    #region Core Calculation Methods

    /// <summary>
    /// Determines which metrics and intervals are valid for the current model type.
    /// </summary>
    private void DetermineValidMetricsAndIntervals()
    {
        // Determine valid metrics
        var allMetricTypes = Enum.GetValues(typeof(MetricType))
                                .Cast<MetricType>()
                                .Where(mt => PredictionStats<T>.IsPredictionStatisticMetric(mt));

        foreach (var metricType in allMetricTypes)
        {
            if (ModelTypeHelper.IsValidMetric(ModelType, metricType))
            {
                _validMetrics.Add(metricType);
            }
        }

        // Determine valid intervals based on model type
        var allIntervalTypes = Enum.GetValues(typeof(IntervalType))
                                .Cast<IntervalType>();

        foreach (var intervalType in allIntervalTypes)
        {
            if (IsValidIntervalForModelType(intervalType, ModelType))
            {
                _validIntervals.Add(intervalType);
            }
        }
    }

    /// <summary>
    /// Determines if a metric type is a prediction statistic metric.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is a prediction statistic; otherwise, false.</returns>
    public static bool IsPredictionStatisticMetric(MetricType metricType)
    {
        // Define which metrics are considered prediction statistics
        return metricType switch
        {
            // Prediction interval coverage
            MetricType.PredictionIntervalCoverage => true,

            // Prediction error metrics
            MetricType.MeanPredictionError => true,
            MetricType.MedianPredictionError => true,

            // Regression prediction metrics 
            MetricType.R2 => true,
            MetricType.AdjustedR2 => true,
            MetricType.ExplainedVarianceScore => true,

            // Learning curve
            MetricType.LearningCurve => true,

            // Classification prediction metrics
            MetricType.Accuracy => true,
            MetricType.Precision => true,
            MetricType.Recall => true,
            MetricType.F1Score => true,

            // Correlation metrics
            MetricType.PearsonCorrelation => true,
            MetricType.SpearmanCorrelation => true,
            MetricType.KendallTau => true,

            // Time series metrics
            MetricType.DynamicTimeWarping => true,

            // Distribution metrics
            MetricType.BestDistributionFit => true,

            // Best correlation type
            MetricType.BestCorrelationType => true,

            // Other metrics that could be derived from error metrics but are relevant for predictions
            MetricType.MSE => true,
            MetricType.RMSE => true,
            MetricType.PopulationStandardError => true,
            MetricType.SampleStandardError => true,
            MetricType.RSS => true,
            MetricType.AIC => true,
            MetricType.BIC => true,
            MetricType.AICAlt => true,

            // For any other metric type
            _ => false,
        };
    }

    /// <summary>
    /// Determines if an interval type is valid for a specific prediction type.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <param name="predictionType">The type of prediction.</param>
    /// <returns>True if the interval is valid for the prediction type; otherwise, false.</returns>
    public static bool IsPredictionIntervalType(IntervalType intervalType, PredictionType predictionType)
    {
        // For regression and time series predictions, most interval types can be valid
        if (predictionType == PredictionType.Regression || predictionType == PredictionType.TimeSeries)
        {
            // All interval types except Forecast are valid for regular regression
            if (predictionType == PredictionType.Regression && intervalType == IntervalType.Forecast)
            {
                return false; // Forecast intervals are specifically for time series
            }

            return true; // All other interval types are valid for regression/time series
        }

        // For distribution predictions, most intervals can be valid
        if (predictionType == PredictionType.Distribution)
        {
            return true;
        }

        // For classification predictions, only certain intervals make sense
        if (predictionType == PredictionType.BinaryClassification ||
            predictionType == PredictionType.MulticlassClassification ||
            predictionType == PredictionType.ProbabilisticClassification ||
            predictionType == PredictionType.MultiLabelClassification)
        {
            return intervalType switch
            {
                // These can be adapted for classification probabilities
                IntervalType.Bootstrap => true,
                IntervalType.Percentile => true,
                IntervalType.Credible => predictionType == PredictionType.ProbabilisticClassification,

                // Other interval types generally don't apply to classification
                _ => false,
            };
        }

        // For ranking predictions, intervals generally don't apply
        if (predictionType == PredictionType.Ranking)
        {
            return false;
        }

        // Default fallback
        return false;
    }

    /// <summary>
    /// Determines if an interval type is valid for a specific model type.
    /// </summary>
    private static bool IsValidIntervalForModelType(IntervalType intervalType, ModelType modelType)
    {
        var modelCategory = ModelTypeHelper.GetCategory(modelType);

        return intervalType switch
        {
            // Most intervals are mainly for regression models
            IntervalType.Prediction => modelCategory == ModelCategory.Regression || modelCategory == ModelCategory.TimeSeries,
            IntervalType.Confidence => modelCategory == ModelCategory.Regression || modelCategory == ModelCategory.TimeSeries,
            IntervalType.Tolerance => modelCategory == ModelCategory.Regression || modelCategory == ModelCategory.TimeSeries,

            // Forecast intervals are specifically for time series models
            IntervalType.Forecast => modelCategory == ModelCategory.TimeSeries,

            // These resampling-based intervals can work for most model types
            IntervalType.Bootstrap => true,
            IntervalType.Jackknife => true,
            IntervalType.Percentile => true,

            // Credible intervals are mainly for Bayesian models
            IntervalType.Credible => modelCategory == ModelCategory.Regression || modelCategory == ModelCategory.TimeSeries,

            // Simultaneous prediction intervals are for regression models
            IntervalType.SimultaneousPrediction => modelCategory == ModelCategory.Regression || modelCategory == ModelCategory.TimeSeries,

            // Default case for any unspecified interval type
            _ => false
        };
    }

    /// <summary>
    /// Calculates all metrics and intervals that are valid for the current model type.
    /// </summary>
    private void CalculateValidMetricsAndIntervals(Vector<T> actual, Vector<T> predicted, int learningCurveSteps,
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

            // Check for cancellation
            cancellationToken.ThrowIfCancellationRequested();

            // Report initial progress
            progress?.Report(0);

            // First calculate the distribution fit as it's needed for several other calculations
            if (_validMetrics.Contains(MetricType.BestDistributionFit) ||
                _validIntervals.Contains(IntervalType.Confidence) ||
                _validIntervals.Contains(IntervalType.Credible))
            {
                BestDistributionFit = StatisticsHelper<T>.DetermineBestFitDistribution(predicted);
                progress?.Report(0.05); // 5% progress for distribution fit
            }

            // Calculate basic metrics first - these are often dependency for other metrics
            CalculateBasicMetrics(actual, predicted);
            progress?.Report(0.25); // 25% progress for basic metrics

            // Calculate intervals
            CalculateIntervals(actual, predicted);
            progress?.Report(0.50); // 50% progress for intervals

            // Calculate learning curve if needed (computationally expensive)
            if (_validMetrics.Contains(MetricType.LearningCurve) && learningCurveSteps > 0)
            {
                LearningCurve = StatisticsHelper<T>.CalculateLearningCurve(actual, predicted, learningCurveSteps);
                _calculatedMetrics.Add(MetricType.LearningCurve);
                progress?.Report(0.75); // 75% progress after learning curve
            }

            // Calculate classification metrics if applicable
            if (_predictionType == PredictionType.BinaryClassification ||
                _predictionType == PredictionType.MulticlassClassification)
            {
                CalculateClassificationMetrics(actual, predicted);
            }

            // Check for cancellation before final steps
            cancellationToken.ThrowIfCancellationRequested();

            // Calculate dependent metrics that rely on previously calculated values
            CalculateDependentMetrics(actual.Length);

            // Report completion
            progress?.Report(1.0);
        }
        catch (OperationCanceledException)
        {
            // Propagate cancellation
            throw;
        }
        catch (Exception ex)
        {
            // Wrap in a custom exception with context information
            throw new PredictionStatsException($"Error calculating metrics for model type {ModelType}", ex);
        }
    }

    /// <summary>
    /// Calculates basic metrics that may be needed for other calculations.
    /// </summary>
    private void CalculateBasicMetrics(Vector<T> actual, Vector<T> predicted)
    {
        // Calculate error metrics
        if (_validMetrics.Contains(MetricType.MeanPredictionError))
        {
            _metrics[MetricType.MeanPredictionError] = StatisticsHelper<T>.CalculateMeanPredictionError(actual, predicted);
            _calculatedMetrics.Add(MetricType.MeanPredictionError);
        }

        if (_validMetrics.Contains(MetricType.MedianPredictionError))
        {
            _metrics[MetricType.MedianPredictionError] = StatisticsHelper<T>.CalculateMedianPredictionError(actual, predicted);
            _calculatedMetrics.Add(MetricType.MedianPredictionError);
        }

        // Calculate regression metrics
        if (_validMetrics.Contains(MetricType.R2))
        {
            _metrics[MetricType.R2] = StatisticsHelper<T>.CalculateR2(actual, predicted);
            _calculatedMetrics.Add(MetricType.R2);
        }

        if (_validMetrics.Contains(MetricType.ExplainedVarianceScore))
        {
            _metrics[MetricType.ExplainedVarianceScore] = StatisticsHelper<T>.CalculateExplainedVarianceScore(actual, predicted);
            _calculatedMetrics.Add(MetricType.ExplainedVarianceScore);
        }

        // Calculate correlation metrics
        if (_validMetrics.Contains(MetricType.PearsonCorrelation))
        {
            _metrics[MetricType.PearsonCorrelation] = StatisticsHelper<T>.CalculatePearsonCorrelationCoefficient(actual, predicted);
            _calculatedMetrics.Add(MetricType.PearsonCorrelation);
        }

        if (_validMetrics.Contains(MetricType.SpearmanCorrelation))
        {
            _metrics[MetricType.SpearmanCorrelation] = StatisticsHelper<T>.CalculateSpearmanRankCorrelationCoefficient(actual, predicted);
            _calculatedMetrics.Add(MetricType.SpearmanCorrelation);
        }

        if (_validMetrics.Contains(MetricType.KendallTau))
        {
            _metrics[MetricType.KendallTau] = StatisticsHelper<T>.CalculateKendallTau(actual, predicted);
            _calculatedMetrics.Add(MetricType.KendallTau);
        }

        if (_validMetrics.Contains(MetricType.DynamicTimeWarping))
        {
            _metrics[MetricType.DynamicTimeWarping] = StatisticsHelper<T>.CalculateDynamicTimeWarping(actual, predicted);
            _calculatedMetrics.Add(MetricType.DynamicTimeWarping);
        }
    }

    /// <summary>
    /// Calculates intervals based on the model type.
    /// </summary>
    private void CalculateIntervals(Vector<T> actual, Vector<T> predicted)
    {
        // Calculate prediction interval (needed for coverage)
        if (_validIntervals.Contains(IntervalType.Prediction))
        {
            _intervals[IntervalType.Prediction] = StatisticsHelper<T>.CalculatePredictionIntervals(actual, predicted, _confidenceLevel);
            _calculatedIntervals.Add(IntervalType.Prediction);

            // Calculate prediction interval coverage if the interval is calculated
            if (_validMetrics.Contains(MetricType.PredictionIntervalCoverage))
            {
                var interval = _intervals[IntervalType.Prediction];
                _metrics[MetricType.PredictionIntervalCoverage] = StatisticsHelper<T>.CalculatePredictionIntervalCoverage(
                    actual, predicted, interval.Lower, interval.Upper);
                _calculatedMetrics.Add(MetricType.PredictionIntervalCoverage);
            }
        }

        // Calculate other interval types if valid
        if (_validIntervals.Contains(IntervalType.Confidence))
        {
            _intervals[IntervalType.Confidence] = StatisticsHelper<T>.CalculateConfidenceIntervals(
                predicted, _confidenceLevel, BestDistributionFit.DistributionType);
            _calculatedIntervals.Add(IntervalType.Confidence);
        }

        if (_validIntervals.Contains(IntervalType.Credible))
        {
            _intervals[IntervalType.Credible] = StatisticsHelper<T>.CalculateCredibleIntervals(
                predicted, _confidenceLevel, BestDistributionFit.DistributionType);
            _calculatedIntervals.Add(IntervalType.Credible);
        }

        if (_validIntervals.Contains(IntervalType.Tolerance))
        {
            _intervals[IntervalType.Tolerance] = StatisticsHelper<T>.CalculateToleranceInterval(
                actual, predicted, _confidenceLevel);
            _calculatedIntervals.Add(IntervalType.Tolerance);
        }

        if (_validIntervals.Contains(IntervalType.Forecast) &&
            ModelTypeHelper.GetCategory(ModelType) == ModelCategory.TimeSeries)
        {
            _intervals[IntervalType.Forecast] = StatisticsHelper<T>.CalculateForecastInterval(
                actual, predicted, _confidenceLevel);
            _calculatedIntervals.Add(IntervalType.Forecast);
        }

        if (_validIntervals.Contains(IntervalType.Bootstrap))
        {
            _intervals[IntervalType.Bootstrap] = StatisticsHelper<T>.CalculateBootstrapInterval(
                actual, predicted, _confidenceLevel);
            _calculatedIntervals.Add(IntervalType.Bootstrap);
        }

        if (_validIntervals.Contains(IntervalType.SimultaneousPrediction))
        {
            _intervals[IntervalType.SimultaneousPrediction] = StatisticsHelper<T>.CalculateSimultaneousPredictionInterval(
                actual, predicted, _confidenceLevel);
            _calculatedIntervals.Add(IntervalType.SimultaneousPrediction);
        }

        if (_validIntervals.Contains(IntervalType.Jackknife))
        {
            _intervals[IntervalType.Jackknife] = StatisticsHelper<T>.CalculateJackknifeInterval(
                actual, predicted);
            _calculatedIntervals.Add(IntervalType.Jackknife);
        }

        if (_validIntervals.Contains(IntervalType.Percentile))
        {
            _intervals[IntervalType.Percentile] = StatisticsHelper<T>.CalculatePercentileInterval(
                predicted, _confidenceLevel);
            _calculatedIntervals.Add(IntervalType.Percentile);
        }

        // Calculate quantile intervals if needed
        QuantileIntervals = StatisticsHelper<T>.CalculateQuantileIntervals(
            actual, predicted, [_numOps.FromDouble(0.25), _numOps.FromDouble(0.5), _numOps.FromDouble(0.75)]);
    }

    /// <summary>
    /// Calculates classification-specific metrics.
    /// </summary>
    private void CalculateClassificationMetrics(Vector<T> actual, Vector<T> predicted)
    {
        // Calculate accuracy if valid
        if (_validMetrics.Contains(MetricType.Accuracy))
        {
            _metrics[MetricType.Accuracy] = StatisticsHelper<T>.CalculateAccuracy(actual, predicted, _predictionType);
            _calculatedMetrics.Add(MetricType.Accuracy);
        }

        // Calculate precision, recall, and F1 score if any of them are valid
        if (_validMetrics.Contains(MetricType.Precision) ||
            _validMetrics.Contains(MetricType.Recall) ||
            _validMetrics.Contains(MetricType.F1Score))
        {
            var (precision, recall, f1) = StatisticsHelper<T>.CalculatePrecisionRecallF1(actual, predicted, _predictionType);

            if (_validMetrics.Contains(MetricType.Precision))
            {
                _metrics[MetricType.Precision] = precision;
                _calculatedMetrics.Add(MetricType.Precision);
            }

            if (_validMetrics.Contains(MetricType.Recall))
            {
                _metrics[MetricType.Recall] = recall;
                _calculatedMetrics.Add(MetricType.Recall);
            }

            if (_validMetrics.Contains(MetricType.F1Score))
            {
                _metrics[MetricType.F1Score] = f1;
                _calculatedMetrics.Add(MetricType.F1Score);
            }
        }
    }

    /// <summary>
    /// Calculates metrics that depend on other metrics, ensuring proper calculation order.
    /// </summary>
    /// <param name="n">Number of observations.</param>
    private void CalculateDependentMetrics(int n)
    {
        // AdjustedR2 depends on R2
        if (_calculatedMetrics.Contains(MetricType.R2) &&
            _validMetrics.Contains(MetricType.AdjustedR2) &&
            !_calculatedMetrics.Contains(MetricType.AdjustedR2))
        {
            var r2 = _metrics[MetricType.R2];
            _metrics[MetricType.AdjustedR2] = StatisticsHelper<T>.CalculateAdjustedR2(r2, n, _numberOfParameters);
            _calculatedMetrics.Add(MetricType.AdjustedR2);
        }

        // RMSE depends on MSE (if both are in PredictionStats)
        if (_calculatedMetrics.Contains(MetricType.MSE) &&
            _validMetrics.Contains(MetricType.RMSE) &&
            !_calculatedMetrics.Contains(MetricType.RMSE))
        {
            var mse = _metrics[MetricType.MSE];
            _metrics[MetricType.RMSE] = _numOps.Sqrt(mse);
            _calculatedMetrics.Add(MetricType.RMSE);
        }

        // F1Score can be calculated from Precision and Recall if not already calculated together
        if (_calculatedMetrics.Contains(MetricType.Precision) &&
            _calculatedMetrics.Contains(MetricType.Recall) &&
            _validMetrics.Contains(MetricType.F1Score) &&
            !_calculatedMetrics.Contains(MetricType.F1Score))
        {
            var precision = _metrics[MetricType.Precision];
            var recall = _metrics[MetricType.Recall];

            // F1 = 2 * (precision * recall) / (precision + recall)
            var numerator = _numOps.Multiply(_numOps.FromDouble(2.0), _numOps.Multiply(precision, recall));
            var denominator = _numOps.Add(precision, recall);

            // Handle potential division by zero
            if (_numOps.GreaterThan(denominator, _numOps.Zero))
            {
                _metrics[MetricType.F1Score] = _numOps.Divide(numerator, denominator);
            }
            else
            {
                _metrics[MetricType.F1Score] = _numOps.Zero;
            }

            _calculatedMetrics.Add(MetricType.F1Score);
        }

        // PopulationStandardError can be derived from MSE if available
        if (_calculatedMetrics.Contains(MetricType.MSE) &&
            _validMetrics.Contains(MetricType.PopulationStandardError) &&
            !_calculatedMetrics.Contains(MetricType.PopulationStandardError))
        {
            var mse = _metrics[MetricType.MSE];
            _metrics[MetricType.PopulationStandardError] = _numOps.Sqrt(mse);
            _calculatedMetrics.Add(MetricType.PopulationStandardError);
        }

        // SampleStandardError depends on PopulationStandardError and parameters
        if (_calculatedMetrics.Contains(MetricType.PopulationStandardError) &&
            _validMetrics.Contains(MetricType.SampleStandardError) &&
            !_calculatedMetrics.Contains(MetricType.SampleStandardError) &&
            n > _numberOfParameters)
        {
            var popStdErr = _metrics[MetricType.PopulationStandardError];
            var correctionFactor = _numOps.FromDouble((double)n / (n - _numberOfParameters));
            correctionFactor = _numOps.Sqrt(correctionFactor);
            _metrics[MetricType.SampleStandardError] = _numOps.Multiply(popStdErr, correctionFactor);
            _calculatedMetrics.Add(MetricType.SampleStandardError);
        }

        // If we calculated various correlation coefficients, can determine the strongest correlation type
        if (_calculatedMetrics.Contains(MetricType.PearsonCorrelation) &&
            _calculatedMetrics.Contains(MetricType.SpearmanCorrelation) &&
            _validMetrics.Contains(MetricType.BestCorrelationType) &&
            !_calculatedMetrics.Contains(MetricType.BestCorrelationType))
        {
            var pearson = _numOps.Abs(_metrics[MetricType.PearsonCorrelation]);
            var spearman = _numOps.Abs(_metrics[MetricType.SpearmanCorrelation]);

            // Store the enum value as a numeric representation
            // 1 for Pearson, 2 for Spearman
            if (_numOps.GreaterThan(pearson, spearman))
            {
                _metrics[MetricType.BestCorrelationType] = _numOps.FromDouble(1.0);
            }
            else
            {
                _metrics[MetricType.BestCorrelationType] = _numOps.FromDouble(2.0);
            }

            _calculatedMetrics.Add(MetricType.BestCorrelationType);
        }

        // AIC, BIC, and AICAlt depend on RSS and parameters
        if (_calculatedMetrics.Contains(MetricType.RSS) &&
            n > 0)
        {
            var rss = _metrics[MetricType.RSS];

            // Calculate AIC if valid and not already calculated
            if (_validMetrics.Contains(MetricType.AIC) &&
                !_calculatedMetrics.Contains(MetricType.AIC))
            {
                _metrics[MetricType.AIC] = StatisticsHelper<T>.CalculateAIC(n, _numberOfParameters, rss);
                _calculatedMetrics.Add(MetricType.AIC);
            }

            // Calculate BIC if valid and not already calculated
            if (_validMetrics.Contains(MetricType.BIC) &&
                !_calculatedMetrics.Contains(MetricType.BIC))
            {
                _metrics[MetricType.BIC] = StatisticsHelper<T>.CalculateBIC(n, _numberOfParameters, rss);
                _calculatedMetrics.Add(MetricType.BIC);
            }

            // Calculate alternative AIC if valid and not already calculated
            if (_validMetrics.Contains(MetricType.AICAlt) &&
                !_calculatedMetrics.Contains(MetricType.AICAlt))
            {
                _metrics[MetricType.AICAlt] = StatisticsHelper<T>.CalculateAICAlternative(n, _numberOfParameters, rss);
                _calculatedMetrics.Add(MetricType.AICAlt);
            }
        }
    }

    #endregion

    #region Public API Methods

    /// <summary>
    /// Gets the value of a specific metric.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the requested metric.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the requested metric is not valid for the current model type.</exception>
    /// <remarks>
    /// For Beginners:
    /// This method provides a convenient way to get a specific metric value using the MetricType enum.
    /// 
    /// For example, instead of directly accessing the R2 property, you could use:
    /// <code>
    /// var r2Value = stats.GetMetric(MetricType.R2);
    /// </code>
    /// 
    /// This is particularly useful when you want to programmatically access different metrics
    /// based on user input or configuration settings.
    /// 
    /// If you request a metric type that doesn't exist, the method will throw an InvalidOperationException.
    /// </remarks>
    public T GetMetric(MetricType metricType)
    {
        if (!IsValidMetric(metricType))
        {
            throw new InvalidOperationException($"Metric {metricType} is not valid for model type {ModelType}.");
        }

        // Special handling for LearningCurve since it's a List<T> not a T
        if (metricType == MetricType.LearningCurve)
        {
            throw new InvalidOperationException("Learning curve is not a scalar metric. Access LearningCurve property directly.");
        }

        return _metrics.TryGetValue(metricType, out var value) ? value : _numOps.Zero;
    }

    /// <summary>
    /// Checks if a specific metric is valid for the current model type.
    /// </summary>
    /// <param name="metricType">The type of metric to check.</param>
    /// <returns>True if the metric is valid for the current model type; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether a specific metric type is valid for the current model type.
    /// It uses the ModelTypeHelper to check if the metric is appropriate for the model category.
    /// </para>
    /// <para>
    /// Different metrics are appropriate for different types of models. For example, R-squared is
    /// appropriate for regression models but not for classification models. This method helps
    /// ensure that only appropriate metrics are calculated and used.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to check if a particular metric is valid for your model type
    /// before trying to get its value. It helps prevent errors by letting you know in advance
    /// whether a metric makes sense for your model.
    /// </para>
    /// <para>
    /// For example, asking for "accuracy" when evaluating a regression model wouldn't make sense,
    /// and this method would return false in that case.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Check if R-squared is valid for this model type
    /// if (stats.IsValidMetric(MetricType.R2))
    /// {
    ///     // Use R-squared...
    ///     var r2 = stats.GetMetric(MetricType.R2);
    ///     Console.WriteLine($"R-squared: {r2}");
    /// }
    /// </code>
    /// </example>
    public bool IsValidMetric(MetricType metricType)
    {
        // Special handling for LearningCurve (which is a List<T>, not a T)
        if (metricType == MetricType.LearningCurve)
        {
            return _validMetrics.Contains(metricType);
        }

        // Use ModelTypeHelper to determine if the metric is valid for this model type
        return ModelTypeHelper.IsValidMetric(ModelType, metricType) && _validMetrics.Contains(metricType);
    }

    /// <summary>
    /// Checks if a specific metric has been calculated.
    /// </summary>
    /// <param name="metricType">The type of metric to check.</param>
    /// <returns>True if the metric has been calculated; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether a specific metric has been calculated.
    /// Even if a metric is valid for a model type, it might not have been calculated yet.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method tells you whether a particular metric has been calculated.
    /// It's useful when you want to check if a metric is available before trying to use it.
    /// </para>
    /// <para>
    /// For example, some metrics might not be calculated by default to save processing time,
    /// and this method lets you check if they're available before trying to access them.
    /// </para>
    /// </remarks>
    /// <example>
    /// <code>
    /// // Check if the learning curve has been calculated
    /// if (stats.IsCalculatedMetric(MetricType.LearningCurve))
    /// {
    ///     // Use the learning curve...
    ///     var learningCurve = stats.LearningCurve;
    ///     // Process learning curve data...
    /// }
    /// else
    /// {
    ///     Console.WriteLine("Learning curve not calculated. Try increasing learningCurveSteps parameter.");
    /// }
    /// </code>
    /// </example>
    public bool IsCalculatedMetric(MetricType metricType)
    {
        return _calculatedMetrics.Contains(metricType);
    }

    /// <summary>
    /// Gets an interval by type.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <returns>The interval as a tuple of (Lower, Upper) bounds.</returns>
    /// <exception cref="InvalidOperationException">Thrown when the requested interval is not valid for the current model type.</exception>
    /// <remarks>
    /// <para>
    /// This method provides a way to access prediction intervals by their type. It will return a tuple
    /// containing the lower and upper bounds of the requested interval.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method provides a way to get a specific interval using the IntervalType enum.
    /// 
    /// For example, instead of directly accessing the PredictionInterval property, you could use:
    /// <code>
    /// var interval = stats.GetInterval(IntervalType.Prediction);
    /// </code>
    /// 
    /// This is useful for programmatically accessing different interval types.
    /// </para>
    /// </remarks>
    public (T Lower, T Upper) GetInterval(IntervalType intervalType)
    {
        if (!IsValidInterval(intervalType))
        {
            throw new InvalidOperationException($"Interval {intervalType} is not valid for model type {ModelType}.");
        }

        return _intervals.TryGetValue(intervalType, out var interval) ? interval : (_numOps.Zero, _numOps.Zero);
    }

    /// <summary>
    /// Tries to get a specific interval.
    /// </summary>
    /// <param name="intervalType">The type of interval to retrieve.</param>
    /// <param name="interval">The interval as a tuple of (Lower, Upper) bounds if successful.</param>
    /// <returns>True if the interval was successfully retrieved; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method provides a safe way to access prediction intervals without throwing exceptions.
    /// It returns true if the interval was successfully retrieved, and false otherwise.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a safer version of GetInterval that won't throw an exception if the interval
    /// is invalid or hasn't been calculated. Instead, it returns false and sets the output
    /// interval to (0, 0). This pattern is commonly used in production code to avoid exceptions.
    /// </para>
    /// </remarks>
    public bool TryGetInterval(IntervalType intervalType, out (T Lower, T Upper) interval)
    {
        if (!IsValidInterval(intervalType))
        {
            interval = (_numOps.Zero, _numOps.Zero);
            return false;
        }

        return _intervals.TryGetValue(intervalType, out interval);
    }

    /// <summary>
    /// Checks if a specific interval is valid for the current model type.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval is valid for the current model type; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether a particular type of interval is valid for the current model type.
    /// Different model types support different kinds of intervals.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to check if a particular interval type is valid for your model type
    /// before trying to get its value. For example, forecast intervals only make sense for time
    /// series models, not for general regression models.
    /// </para>
    /// </remarks>
    public bool IsValidInterval(IntervalType intervalType)
    {
        return _validIntervals.Contains(intervalType);
    }

    /// <summary>
    /// Checks if a specific interval has been calculated.
    /// </summary>
    /// <param name="intervalType">The type of interval to check.</param>
    /// <returns>True if the interval has been calculated; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method determines whether a particular type of interval has been calculated.
    /// It can be used to avoid requesting intervals that haven't been calculated yet.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method tells you whether a particular interval has been 
    /// calculated. It's useful when you want to check if an interval is available before trying to use it.
    /// </para>
    /// </remarks>
    public bool IsCalculatedInterval(IntervalType intervalType)
    {
        return _calculatedIntervals.Contains(intervalType);
    }

    /// <summary>
    /// Gets all interval types that are valid for the current model type.
    /// </summary>
    /// <returns>An array of valid interval types.</returns>
    /// <remarks>
    /// <para>
    /// This method returns all interval types that are valid for the current model type.
    /// It can be used to determine which intervals are available for the current model.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method gives you a list of all interval types that make sense for your model type.
    /// Different model types support different kinds of intervals, so this helps you
    /// understand which ones are available for your specific model.
    /// </para>
    /// </remarks>
    public IntervalType[] GetValidIntervalTypes()
    {
        return [.. _validIntervals];
    }

    /// <summary>
    /// Gets all interval types that have been calculated.
    /// </summary>
    /// <returns>An array of calculated interval types.</returns>
    /// <remarks>
    /// <para>
    /// This method returns all interval types that have been calculated.
    /// It can be used to determine which intervals are available for use.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method gives you a list of all interval types that have been calculated.
    /// It's useful when you want to know which intervals are actually available for use,
    /// rather than just which ones are valid for your model type.
    /// </para>
    /// </remarks>
    public IntervalType[] GetCalculatedIntervalTypes()
    {
        return [.. _calculatedIntervals];
    }

    /// <summary>
    /// Gets a dictionary of all calculated intervals.
    /// </summary>
    /// <returns>A dictionary mapping interval types to their values.</returns>
    /// <remarks>
    /// <para>
    /// This method returns a dictionary of all calculated intervals.
    /// It can be used to access all intervals at once.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This method gives you all calculated intervals in one go. It's useful when you
    /// want to work with multiple intervals at once, perhaps to compare them or display
    /// them in a report.
    /// </para>
    /// </remarks>
    public Dictionary<IntervalType, (T Lower, T Upper)> GetAllCalculatedIntervals()
    {
        return new Dictionary<IntervalType, (T Lower, T Upper)>(_intervals);
    }

    #endregion
}