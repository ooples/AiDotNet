namespace AiDotNet.Statistics;

/// <summary>
/// Calculates and stores various statistics to evaluate prediction performance and generate prediction intervals.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class provides a comprehensive set of metrics to evaluate predictive models and 
/// calculate different types of statistical intervals around predictions.
/// </para>
/// 
/// <para>
/// For Beginners:
/// When you build a predictive model (like a machine learning model), you often want to:
/// 1. Measure how well your model performs (using metrics like R-squared (R2), accuracy, etc.)
/// 2. Understand how confident you can be in your predictions (using various intervals)
/// 3. Understand the relationship between actual and predicted values (using correlations)
/// 
/// This class helps you do all of these things. The "T" in PredictionStats&lt;T&gt; means it 
/// works with different number types like decimal, double, or float without needing separate 
/// implementations for each.
/// </para>
/// </remarks>
public class PredictionStats<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

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
    private (T Lower, T Upper) _predictionInterval = default;
    public (T Lower, T Upper) PredictionInterval { get { EnsureFullStatsComputed(); return _predictionInterval; } private set { _predictionInterval = value; } }

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
    private (T Lower, T Upper) _confidenceInterval = default;
    public (T Lower, T Upper) ConfidenceInterval { get { EnsureFullStatsComputed(); return _confidenceInterval; } private set { _confidenceInterval = value; } }

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
    private (T Lower, T Upper) _credibleInterval = default;
    public (T Lower, T Upper) CredibleInterval { get { EnsureFullStatsComputed(); return _credibleInterval; } private set { _credibleInterval = value; } }

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
    private (T Lower, T Upper) _toleranceInterval = default;
    public (T Lower, T Upper) ToleranceInterval { get { EnsureFullStatsComputed(); return _toleranceInterval; } private set { _toleranceInterval = value; } }

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
    private (T Lower, T Upper) _forecastInterval = default;
    public (T Lower, T Upper) ForecastInterval { get { EnsureFullStatsComputed(); return _forecastInterval; } private set { _forecastInterval = value; } }

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
    private (T Lower, T Upper) _bootstrapInterval = default;
    public (T Lower, T Upper) BootstrapInterval { get { EnsureFullStatsComputed(); return _bootstrapInterval; } private set { _bootstrapInterval = value; } }

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
    private (T Lower, T Upper) _simultaneousPredictionInterval = default;
    public (T Lower, T Upper) SimultaneousPredictionInterval { get { EnsureFullStatsComputed(); return _simultaneousPredictionInterval; } private set { _simultaneousPredictionInterval = value; } }

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
    private (T Lower, T Upper) _jackknifeInterval = default;
    public (T Lower, T Upper) JackknifeInterval { get { EnsureFullStatsComputed(); return _jackknifeInterval; } private set { _jackknifeInterval = value; } }

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
    private (T Lower, T Upper) _percentileInterval = default;
    public (T Lower, T Upper) PercentileInterval { get { EnsureFullStatsComputed(); return _percentileInterval; } private set { _percentileInterval = value; } }

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
    private List<(T Quantile, T Lower, T Upper)> _quantileIntervals = new();
    public List<(T Quantile, T Lower, T Upper)> QuantileIntervals { get { EnsureFullStatsComputed(); return _quantileIntervals; } private set { _quantileIntervals = value; } }

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
    private T _predictionIntervalCoverage = default!;
    public T PredictionIntervalCoverage { get { EnsureFullStatsComputed(); return _predictionIntervalCoverage; } private set { _predictionIntervalCoverage = value; } }

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
    private T _meanPredictionError = default!;
    public T MeanPredictionError { get { EnsureFullStatsComputed(); return _meanPredictionError; } private set { _meanPredictionError = value; } }

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
    private T _medianPredictionError = default!;
    public T MedianPredictionError { get { EnsureFullStatsComputed(); return _medianPredictionError; } private set { _medianPredictionError = value; } }

    /// <summary>
    /// Coefficient of determination - The proportion of variance in the dependent variable explained by the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// R-squared (R2) is perhaps the most common metric for regression models. It ranges from 0 to 1:
    /// R² (R-squared) is perhaps the most common metric for regression models. It ranges from 0 to 1:
    /// - 1 means your model perfectly predicts all values
    /// - 0 means your model does no better than simply predicting the average for every case
    /// - Values in between indicate the percentage of variance your model explains
    ///
    /// For example, an R2 of 0.75 means your model explains 75% of the variability in the target variable.
    ///
    /// Be careful: a high R2 doesn't necessarily mean your model is good - it could be overfitting!
    /// For example, an R² of 0.75 means your model explains 75% of the variability in the target variable.
    ///
    /// Be careful: a high R² doesn't necessarily mean your model is good - it could be overfitting!
    /// </remarks>
    public T R2 { get; private set; }

    /// <summary>
    /// R-Squared - Alias for R2 property (Coefficient of determination).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is an alternative name for the R2 property, providing the same value.
    /// Some frameworks and documentation use "RSquared" while others use "R2".
    /// Both refer to the proportion of variance in the dependent variable explained by the model.
    /// </remarks>
    public T RSquared => R2;

    /// <summary>
    /// R-squared (R2) adjusted for the number of predictors in the model.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AdjustedR2 is a modified version of R2 that accounts for the number of features in your model.
    /// 
    /// Regular R2 always increases when you add more features, even if they don't actually improve predictions.
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
    /// Like R2, values closer to 1 are better.
    /// Like R², values closer to 1 are better.
    /// </remarks>
    public T AdjustedR2 { get; private set; }

    /// <summary>
    /// The explained variance score - A measure of how well the model accounts for the variance in the data.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// ExplainedVarianceScore is similar to R2, but it doesn't penalize the model for systematic bias.
    /// ExplainedVarianceScore is similar to R², but it doesn't penalize the model for systematic bias.
    /// 
    /// It ranges from 0 to 1, with higher values being better:
    /// - 1 means your model explains all the variance in the data (perfect)
    /// - 0 means your model doesn't explain any variance
    /// 
    /// If your model's predictions are all shifted by a constant amount from the actual values,
    /// R2 would be lower, but ExplainedVarianceScore would still be high.
    /// R² would be lower, but ExplainedVarianceScore would still be high.
    /// </remarks>
    private T _explainedVarianceScore = default!;
    public T ExplainedVarianceScore { get { EnsureFullStatsComputed(); return _explainedVarianceScore; } private set { _explainedVarianceScore = value; } }

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
    private List<T> _learningCurve = new();
    public List<T> LearningCurve { get { EnsureFullStatsComputed(); return _learningCurve; } private set { _learningCurve = value; } }

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
    private T _accuracy = default!;
    public T Accuracy { get { EnsureFullStatsComputed(); return _accuracy; } private set { _accuracy = value; } }

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
    private T _precision = default!;
    public T Precision { get { EnsureFullStatsComputed(); return _precision; } private set { _precision = value; } }

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
    private T _recall = default!;
    public T Recall { get { EnsureFullStatsComputed(); return _recall; } private set { _recall = value; } }

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
    private T _f1Score = default!;
    public T F1Score { get { EnsureFullStatsComputed(); return _f1Score; } private set { _f1Score = value; } }

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
    private T _pearsonCorrelation = default!;
    public T PearsonCorrelation { get { EnsureFullStatsComputed(); return _pearsonCorrelation; } private set { _pearsonCorrelation = value; } }

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
    private T _spearmanCorrelation = default!;
    public T SpearmanCorrelation { get { EnsureFullStatsComputed(); return _spearmanCorrelation; } private set { _spearmanCorrelation = value; } }

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
    private T _kendallTau = default!;
    public T KendallTau { get { EnsureFullStatsComputed(); return _kendallTau; } private set { _kendallTau = value; } }

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
    private T _dynamicTimeWarping = default!;
    public T DynamicTimeWarping { get { EnsureFullStatsComputed(); return _dynamicTimeWarping; } private set { _dynamicTimeWarping = value; } }

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
    private DistributionFitResult<T> _bestDistributionFit = new();
    public DistributionFitResult<T> BestDistributionFit { get { EnsureFullStatsComputed(); return _bestDistributionFit; } private set { _bestDistributionFit = value; } }

    /// <summary>
    /// Creates a new PredictionStats instance and calculates all prediction metrics.
    /// </summary>
    /// <param name="inputs">The inputs containing actual values, predicted values, and calculation parameters.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor takes your actual values (ground truth), predicted values from your model,
    /// and additional parameters, then calculates all the prediction metrics in one step.
    /// 
    /// The inputs object includes:
    /// - Actual: The true values you're trying to predict
    /// - Predicted: Your model's predictions
    /// - ConfidenceLevel: The probability level for intervals (typically 0.95 for 95% confidence)
    /// - NumberOfParameters: How many features your model uses
    /// - Other settings that control specific calculations
    /// </remarks>
    private PredictionStatsInputs<T>? _deferredInputs;
    private bool _fullStatsComputed;

    internal PredictionStats(PredictionStatsInputs<T> inputs)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize all properties to zero
        PredictionInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        ConfidenceInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        CredibleInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        ToleranceInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        ForecastInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        BootstrapInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        SimultaneousPredictionInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        JackknifeInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        PercentileInterval = (Lower: _numOps.Zero, Upper: _numOps.Zero);
        QuantileIntervals = new List<(T Quantile, T Lower, T Upper)>();
        PredictionIntervalCoverage = _numOps.Zero;
        MeanPredictionError = _numOps.Zero;
        MedianPredictionError = _numOps.Zero;
        R2 = _numOps.Zero;
        AdjustedR2 = _numOps.Zero;
        PearsonCorrelation = _numOps.Zero;
        SpearmanCorrelation = _numOps.Zero;
        KendallTau = _numOps.Zero;
        DynamicTimeWarping = _numOps.Zero;
        ExplainedVarianceScore = _numOps.Zero;
        LearningCurve = new List<T>();
        Accuracy = _numOps.Zero;
        Precision = _numOps.Zero;
        Recall = _numOps.Zero;
        F1Score = _numOps.Zero;

        if (inputs.Actual.Length > 0 && inputs.Predicted.Length > 0)
        {
            if (inputs.Actual.Length != inputs.Predicted.Length)
                throw new ArgumentException("Actual and predicted vectors must have the same length.", nameof(inputs));

            // Compute ONLY R² and AdjustedR² eagerly — these are what the optimizer
            // fitness calculator needs. All other stats are deferred until first access.
            R2 = StatisticsHelper<T>.CalculateR2(inputs.Actual, inputs.Predicted);
            AdjustedR2 = StatisticsHelper<T>.CalculateAdjustedR2(R2, inputs.Actual.Length, inputs.NumberOfParameters);

            // Store inputs for lazy computation of the remaining 30+ metrics
            _deferredInputs = inputs;
        }
    }

    /// <summary>
    /// Ensures all stats beyond R²/AdjustedR² are computed. Called lazily on first access
    /// to any property other than R2/AdjustedR2.
    /// </summary>
    private void EnsureFullStatsComputed()
    {
        if (_fullStatsComputed || _deferredInputs is null) return;

        // Run the heavy computation BEFORE marking the instance "computed" so that an
        // exception inside CalculatePredictionStats leaves the lazy state intact —
        // the next property access will retry instead of silently returning default
        // values forever (the corruption-on-throw bug noted in PR review).
        CalculatePredictionStats(_deferredInputs.Actual, _deferredInputs.Predicted,
            _deferredInputs.NumberOfParameters, _numOps.FromDouble(_deferredInputs.ConfidenceLevel),
            _deferredInputs.LearningCurveSteps, _deferredInputs.PredictionType);

        _fullStatsComputed = true;
        _deferredInputs = null; // Release reference
    }

    /// <summary>
    /// Creates an empty PredictionStats instance with all metrics set to zero.
    /// </summary>
    /// <returns>A PredictionStats instance with all metrics initialized to zero.</returns>
    /// <remarks>
    /// For Beginners:
    /// This static method creates a PredictionStats object where all metrics are set to zero.
    /// It's useful when you need a placeholder or default instance, or when you want to
    /// compare against a baseline of "perfect predictions."
    /// </remarks>
    public static PredictionStats<T> Empty()
    {
        return new PredictionStats<T>(new());
    }

    /// <summary>
    /// Creates a lightweight PredictionStats with only the R² coefficient populated.
    /// All other metrics (AdjustedR², MAE, RMSE, learning curves, confidence intervals,
    /// etc.) remain at their default zero values.
    /// </summary>
    /// <param name="r2">Pre-computed coefficient of determination.</param>
    /// <returns>A PredictionStats instance with R² set, ready to satisfy fit-detection
    /// checks without paying the cost of a full statistics computation.</returns>
    /// <remarks>
    /// <para>
    /// This factory exists so the optimizer can compute a cheap R² for non-preferred
    /// datasets (the ones the FitnessCalculator doesn't score against) and still feed
    /// FitDetector — which reads <c>TrainingSet.PredictionStats.R2</c>, etc. — without
    /// running the full <c>PredictionStats</c> constructor (learning curves, confidence
    /// intervals, etc.) for every non-preferred dataset on every epoch.
    /// </para>
    /// <para><b>For Beginners:</b> This is a "just enough" version of PredictionStats
    /// for code paths that only need R² (the most common single metric used to detect
    /// over- and underfitting). It avoids the heavier metric calculations.
    /// </para>
    /// </remarks>
    public static PredictionStats<T> WithR2Only(T r2)
    {
        var stats = new PredictionStats<T>(new());
        stats.R2 = r2;
        return stats;
    }

    /// <summary>
    /// Calculates all prediction statistics based on actual and predicted values.
    /// </summary>
    /// <param name="actual">Vector of actual values (ground truth).</param>
    /// <param name="predicted">Vector of predicted values from your model.</param>
    /// <param name="numberOfParameters">Number of features or parameters in your model.</param>
    /// <param name="confidenceLevel">The confidence level for statistical intervals (e.g., 0.95 for 95% confidence).</param>
    /// <param name="learningCurveSteps">Number of steps to use when calculating the learning curve.</param>
    /// <param name="predictionType">The type of prediction (regression, binary classification, etc.).</param>
    /// <remarks>
    /// For Beginners:
    /// This private method does the actual work of calculating all the prediction statistics.
    /// 
    /// - actual: These are the true values you're trying to predict
    /// - predicted: These are your model's predictions
    /// - numberOfParameters: This is how many input features your model uses, which is needed for some metrics
    /// - confidenceLevel: This determines how wide your intervals are (e.g., 0.95 gives 95% confidence intervals)
    /// - learningCurveSteps: This controls how many points are calculated in the learning curve
    /// - predictionType: This tells the method whether you're doing regression (predicting numbers) or classification (predicting categories)
    /// 
    /// The method calculates various metrics like intervals, correlation coefficients, and performance metrics,
    /// storing the results in the corresponding properties.
    /// </remarks>
    private void CalculatePredictionStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, T confidenceLevel, int learningCurveSteps, PredictionType predictionType)
    {
        // All intermediate values computed into locals; properties are set
        // together at the bottom. Before this rewrite the method read
        // AdjustedR2's dependency via R2's property getter, which calls
        // EnsureFullStatsComputed and re-enters this method — unbounded
        // recursion + StackOverflowException in the test host. Same class
        // of bug as BasicStats / ErrorStats; see those files for context.
        var bestDist = StatisticsHelper<T>.DetermineBestFitDistribution(predicted);

        T meanPredErr = StatisticsHelper<T>.CalculateMeanPredictionError(actual, predicted);
        T medianPredErr = StatisticsHelper<T>.CalculateMedianPredictionError(actual, predicted);

        // R2 and AdjustedR2 were already computed eagerly in the ctor
        // (see lines 570-571) with the same Actual/Predicted/NumberOfParameters
        // inputs. Reuse the stored values instead of calling StatisticsHelper
        // a second time — cuts a redundant O(n) scan per metric over the
        // prediction vector on the lazy-compute path.
        T r2 = R2;
        T adjR2 = AdjustedR2;
        T evs = StatisticsHelper<T>.CalculateExplainedVarianceScore(actual, predicted);
        var lc = StatisticsHelper<T>.CalculateLearningCurve(actual, predicted, learningCurveSteps);
        T pearson = StatisticsHelper<T>.CalculatePearsonCorrelationCoefficient(actual, predicted);
        T spearman = StatisticsHelper<T>.CalculateSpearmanRankCorrelationCoefficient(actual, predicted);
        T kendall = StatisticsHelper<T>.CalculateKendallTau(actual, predicted);
        T dtw = StatisticsHelper<T>.CalculateDynamicTimeWarping(actual, predicted);

        (T Lower, T Upper) predInterval = StatisticsHelper<T>.CalculatePredictionIntervals(actual, predicted, confidenceLevel);
        T predIntervalCov = StatisticsHelper<T>.CalculatePredictionIntervalCoverage(actual, predicted, predInterval.Lower, predInterval.Upper);
        var confInterval = StatisticsHelper<T>.CalculateConfidenceIntervals(predicted, confidenceLevel, bestDist.DistributionType);
        var credInterval = StatisticsHelper<T>.CalculateCredibleIntervals(predicted, confidenceLevel, bestDist.DistributionType);
        var tolInterval = StatisticsHelper<T>.CalculateToleranceInterval(actual, predicted, confidenceLevel);
        var forecastInterval = StatisticsHelper<T>.CalculateForecastInterval(actual, predicted, confidenceLevel);
        var quantiles = StatisticsHelper<T>.CalculateQuantileIntervals(actual, predicted, new T[] { _numOps.FromDouble(0.25), _numOps.FromDouble(0.5), _numOps.FromDouble(0.75) });
        var bootstrap = StatisticsHelper<T>.CalculateBootstrapInterval(actual, predicted, confidenceLevel);
        var simulPred = StatisticsHelper<T>.CalculateSimultaneousPredictionInterval(actual, predicted, confidenceLevel);
        var jackknife = StatisticsHelper<T>.CalculateJackknifeInterval(actual, predicted);
        var percentile = StatisticsHelper<T>.CalculatePercentileInterval(predicted, confidenceLevel);

        T accuracy = StatisticsHelper<T>.CalculateAccuracy(actual, predicted, predictionType);
        var (precision, recall, f1) = StatisticsHelper<T>.CalculatePrecisionRecallF1(actual, predicted, predictionType);

        BestDistributionFit = bestDist;
        MeanPredictionError = meanPredErr;
        MedianPredictionError = medianPredErr;
        R2 = r2;
        AdjustedR2 = adjR2;
        ExplainedVarianceScore = evs;
        LearningCurve = lc;
        PearsonCorrelation = pearson;
        SpearmanCorrelation = spearman;
        KendallTau = kendall;
        DynamicTimeWarping = dtw;
        PredictionInterval = predInterval;
        PredictionIntervalCoverage = predIntervalCov;
        ConfidenceInterval = confInterval;
        CredibleInterval = credInterval;
        ToleranceInterval = tolInterval;
        ForecastInterval = forecastInterval;
        QuantileIntervals = quantiles;
        BootstrapInterval = bootstrap;
        SimultaneousPredictionInterval = simulPred;
        JackknifeInterval = jackknife;
        PercentileInterval = percentile;
        Accuracy = accuracy;
        Precision = precision;
        Recall = recall;
        F1Score = f1;
    }

    /// <summary>
    /// Retrieves a specific metric value by metric type.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the requested metric.</returns>
    /// <exception cref="ArgumentException">Thrown when an unknown metric type is requested.</exception>
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
    /// If you request a metric type that doesn't exist, the method will throw an ArgumentException.
    /// </remarks>
    public T GetMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.R2 => R2,
            MetricType.AdjustedR2 => AdjustedR2,
            MetricType.ExplainedVarianceScore => ExplainedVarianceScore,
            MetricType.MeanPredictionError => MeanPredictionError,
            MetricType.MedianPredictionError => MedianPredictionError,
            MetricType.PredictionIntervalCoverage => PredictionIntervalCoverage,
            MetricType.Accuracy => Accuracy,
            MetricType.Precision => Precision,
            MetricType.Recall => Recall,
            MetricType.F1Score => F1Score,
            MetricType.PearsonCorrelation => PearsonCorrelation,
            MetricType.SpearmanCorrelation => SpearmanCorrelation,
            _ => throw new ArgumentException($"Metric {metricType} is not available in PredictionStats.", nameof(metricType)),
        };
    }

    /// <summary>
    /// Checks if a specific metric is available in this PredictionStats instance.
    /// </summary>
    /// <param name="metricType">The type of metric to check for.</param>
    /// <returns>True if the metric is available, false otherwise.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method allows you to check if a particular metric is available before trying to get its value.
    /// It's useful when you're not sure if a specific metric was calculated for this set of predictions.
    /// 
    /// For example:
    /// <code>
    /// if (stats.HasMetric(MetricType.R2))
    /// {
    ///     var r2Value = stats.GetMetric(MetricType.R2);
    ///     // Use r2Value...
    /// }
    /// </code>
    /// 
    /// This prevents errors that might occur if you try to access a metric that wasn't calculated.
    /// </remarks>
    public bool HasMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.R2 => true,
            MetricType.AdjustedR2 => true,
            MetricType.ExplainedVarianceScore => true,
            MetricType.MeanPredictionError => true,
            MetricType.MedianPredictionError => true,
            MetricType.PredictionIntervalCoverage => true,
            MetricType.Accuracy => true,
            MetricType.Precision => true,
            MetricType.Recall => true,
            MetricType.F1Score => true,
            MetricType.PearsonCorrelation => true,
            MetricType.SpearmanCorrelation => true,
            _ => false,
        };
    }
}
