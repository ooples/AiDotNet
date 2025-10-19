﻿namespace AiDotNet.Statistics;

/// <summary>
/// Calculates and stores various error metrics for evaluating prediction model performance.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double, decimal).</typeparam>
/// <remarks>
/// <para>
/// This class provides a comprehensive set of error metrics to assess how well predicted values 
/// match actual values.
/// </para>
/// 
/// <para>
/// For Beginners:
/// When building AI or machine learning models, you need ways to measure how accurate your predictions are.
/// Think of these metrics like different ways to score a test:
/// - Some look at the average error (MAE, MSE)
/// - Some look at percentage differences (MAPE, SMAPE)
/// - Some help detect if your model is consistently overestimating or underestimating (MeanBiasError)
/// - Some are specialized for specific types of predictions (AUCROC, AUCPR for classification)
/// 
/// The "T" in ErrorStats&lt;T&gt; means this class works with different number types like decimal, 
/// double, or float without needing separate implementations for each.
/// </para>
/// </remarks>
public class ErrorStats<T>
{
    /// <summary>
    /// Provides mathematical operations for the generic type T.
    /// </summary>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Mean Absolute Error - The average absolute difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MAE measures the average size of errors without considering their direction (positive or negative).
    /// Lower values indicate better accuracy. If MAE = 5, your predictions are off by 5 units on average.
    /// </remarks>
    public T MAE { get; private set; }

    /// <summary>
    /// Mean Squared Error - The average of squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MSE squares the errors before averaging them, which penalizes large errors more heavily than small ones.
    /// Lower values indicate better accuracy. Because of squaring, the value is not in the same units as your data.
    /// </remarks>
    public T MSE { get; private set; }

    /// <summary>
    /// Root Mean Squared Error - The square root of the Mean Squared Error.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// RMSE converts MSE back to the original units of your data by taking the square root.
    /// It's often preferred over MSE for interpretation because it's in the same units as your data.
    /// Like MAE, lower values indicate better accuracy.
    /// </remarks>
    public T RMSE { get; private set; }

    /// <summary>
    /// Mean Absolute Percentage Error - The average percentage difference between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MAPE expresses error as a percentage, which helps you understand the relative size of errors.
    /// MAPE = 10 means that, on average, your predictions are off by 10% from the actual values.
    /// Note: MAPE can be problematic when actual values are close to zero.
    /// </remarks>
    public T MAPE { get; private set; }

    /// <summary>
    /// Mean Bias Error - The average of prediction errors (predicted - actual).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MeanBiasError helps determine if your model tends to overestimate (positive value) or
    /// underestimate (negative value). Ideally, it should be close to zero, indicating no systematic bias.
    /// Unlike MAE, this doesn't take the absolute value, so positive and negative errors can cancel out.
    /// </remarks>
    public T MeanBiasError { get; private set; }

    /// <summary>
    /// Median Absolute Error - The middle value of all absolute differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Unlike MAE which uses the average, MedianAbsoluteError uses the middle value of all absolute errors.
    /// This makes it less sensitive to outliers (extreme errors) than MAE.
    /// For example, if you have errors of [1, 2, 100], the median is 2, while the mean would be 34.3.
    /// </remarks>
    public T MedianAbsoluteError { get; private set; }

    /// <summary>
    /// Maximum Error - The largest absolute difference between any predicted and actual value.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// MaxError tells you the worst-case error in your predictions.
    /// It can help you understand the potential maximum impact of prediction errors.
    /// </remarks>
    public T MaxError { get; private set; }

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
    public T TheilUStatistic { get; private set; }

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
    public T DurbinWatsonStatistic { get; private set; }

    /// <summary>
    /// Sample Standard Error - An estimate of the standard deviation of prediction errors, adjusted for model complexity.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// SampleStandardError estimates how much prediction errors typically vary, taking into account
    /// how many parameters (features) your model uses. It's useful for constructing confidence intervals
    /// around predictions and is adjusted downward based on the number of parameters in your model.
    /// </remarks>
    public T SampleStandardError { get; private set; }

    /// <summary>
    /// Population Standard Error - The standard deviation of prediction errors without adjustment for model complexity.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// PopulationStandardError measures how much prediction errors typically vary, but unlike 
    /// SampleStandardError, it doesn't adjust for model complexity. It gives you an idea of the
    /// typical size of the errors your model makes.
    /// </remarks>
    public T PopulationStandardError { get; private set; }

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
    public T AIC { get; private set; }

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
    public T BIC { get; private set; }

    /// <summary>
    /// Alternative Akaike Information Criterion - A variant of AIC with a different penalty term.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// AICAlt is another version of AIC that uses a slightly different approach to penalize
    /// model complexity. It's particularly useful when sample sizes are small.
    /// Like AIC and BIC, lower values are better.
    /// </remarks>
    public T AICAlt { get; private set; }

    /// <summary>
    /// Residual Sum of Squares - The sum of squared differences between predicted and actual values.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// RSS is the total squared error of your model. It's the basis for many other metrics like MSE
    /// (which is just RSS divided by the number of observations).
    /// Lower values indicate a better fit. It's used in calculating metrics like AIC and BIC.
    /// </remarks>
    public T RSS { get; private set; }

    /// <summary>
    /// List of individual prediction errors (residuals).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// ErrorList contains the difference between each predicted value and the corresponding actual value.
    /// This lets you examine individual errors, create visualizations like histograms,
    /// or perform additional analyses beyond summary statistics.
    /// </remarks>
    public List<T> ErrorList { get; private set; } = [];

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
    public T AUCPR { get; private set; }

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
    public T AUCROC { get; private set; }

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
    public T SMAPE { get; private set; }

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
    public T MeanSquaredLogError { get; private set; }

    /// <summary>
    /// Mean Absolute Error - Alias for MAE property.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is an alternative name for the MAE property, providing the same value.
    /// Some frameworks and documentation prefer the full name "MeanAbsoluteError" while others use "MAE".
    /// Both refer to the average absolute difference between predicted and actual values.
    /// </remarks>
    public T MeanAbsoluteError => MAE;

    /// <summary>
    /// Mean Squared Error - Alias for MSE property.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is an alternative name for the MSE property, providing the same value.
    /// Some frameworks and documentation prefer the full name "MeanSquaredError" while others use "MSE".
    /// Both refer to the average of squared differences between predicted and actual values.
    /// </remarks>
    public T MeanSquaredError => MSE;

    /// <summary>
    /// Root Mean Squared Error - Alias for RMSE property.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is an alternative name for the RMSE property, providing the same value.
    /// Some frameworks and documentation prefer the full name "RootMeanSquaredError" while others use "RMSE".
    /// Both refer to the square root of the Mean Squared Error.
    /// </remarks>
    public T RootMeanSquaredError => RMSE;

    /// <summary>
    /// Area Under the Curve (ROC) - Alias for AUCROC property.
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// This is an alternative name for the AUCROC property, providing the same value.
    /// In many contexts, "AUC" specifically refers to the area under the ROC curve.
    /// This metric is commonly used to evaluate classification models.
    /// </remarks>
    public T AUC => AUCROC;

    /// <summary>
    /// Classification accuracy - The proportion of correct predictions (for classification tasks).
    /// </summary>
    /// <remarks>
    /// For Beginners:
    /// Accuracy is a simple metric for classification problems. It's the percentage of predictions
    /// that match the actual values.
    ///
    /// For example, if your model correctly classifies 90 out of 100 samples, the accuracy is 0.9 or 90%.
    ///
    /// Note: This property is typically used for classification tasks. For regression tasks,
    /// other metrics like MAE, MSE, or R² are more appropriate.
    ///
    /// While intuitive, accuracy can be misleading for imbalanced classes. For example, if 95% of your
    /// data belongs to class A, a model that always predicts class A would have 95% accuracy
    /// despite being useless for class B.
    /// </remarks>
    public T Accuracy { get; private set; }

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
    public T Precision { get; private set; }

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
    public T Recall { get; private set; }

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
    public T F1Score { get; private set; }

    /// <summary>
    /// Creates a new ErrorStats instance and calculates all error metrics.
    /// </summary>
    /// <param name="inputs">The inputs containing actual and predicted values.</param>
    /// <remarks>
    /// For Beginners:
    /// This constructor takes your actual values (ground truth) and predicted values,
    /// then calculates all the error metrics in one step.
    /// </remarks>
    internal ErrorStats(ErrorStatsInputs<T> inputs)
    {
        _numOps = MathHelper.GetNumericOperations<T>();

        // Initialize all variables to zero
        MAE = _numOps.Zero;
        MSE = _numOps.Zero;
        RMSE = _numOps.Zero;
        MAPE = _numOps.Zero;
        MeanBiasError = _numOps.Zero;
        MedianAbsoluteError = _numOps.Zero;
        MeanSquaredLogError = _numOps.Zero;
        MaxError = _numOps.Zero;
        TheilUStatistic = _numOps.Zero;
        DurbinWatsonStatistic = _numOps.Zero;
        SampleStandardError = _numOps.Zero;
        PopulationStandardError = _numOps.Zero;
        AIC = _numOps.Zero;
        BIC = _numOps.Zero;
        AICAlt = _numOps.Zero;
        RSS = _numOps.Zero;
        AUCPR = _numOps.Zero;
        AUCROC = _numOps.Zero;
        SMAPE = _numOps.Zero;
        Accuracy = _numOps.Zero;
        Precision = _numOps.Zero;
        Recall = _numOps.Zero;
        F1Score = _numOps.Zero;

        ErrorList = [];

        CalculateErrorStats(inputs.Actual, inputs.Predicted, inputs.FeatureCount, inputs.PredictionType);
    }

    /// <summary>
    /// Creates an empty ErrorStats instance with all metrics set to zero.
    /// </summary>
    /// <returns>An ErrorStats instance with all metrics initialized to zero.</returns>
    /// <remarks>
    /// For Beginners:
    /// This static method creates an ErrorStats object where all metrics are set to zero.
    /// It's useful when you need a placeholder or default instance, or when you want to
    /// compare against a baseline of "no errors."
    /// </remarks>
    public static ErrorStats<T> Empty()
    {
        return new ErrorStats<T>(new());
    }

    /// <summary>
    /// Calculates all error metrics based on actual and predicted values.
    /// </summary>
    /// <param name="actual">Vector of actual values (ground truth).</param>
    /// <param name="predicted">Vector of predicted values from your model.</param>
    /// <param name="numberOfParameters">Number of features or parameters in your model.</param>
    /// <param name="predictionType">The type of prediction task (regression or classification).</param>
    /// <remarks>
    /// For Beginners:
    /// This private method does the actual work of calculating all the error metrics.
    ///
    /// - actual: These are the true values you're trying to predict
    /// - predicted: These are your model's predictions
    /// - numberOfParameters: This is how many input features your model uses, which is needed
    ///   for metrics that account for model complexity (like AIC, BIC)
    /// - predictionType: Whether this is a regression or classification task
    ///
    /// The method calculates each error metric using specialized helper methods and
    /// stores the results in the corresponding properties.
    /// </remarks>
    private void CalculateErrorStats(Vector<T> actual, Vector<T> predicted, int numberOfParameters, PredictionType predictionType = PredictionType.Regression)
    {
        int n = actual.Length;

        // Calculate basic error metrics
        MAE = StatisticsHelper<T>.CalculateMeanAbsoluteError(actual, predicted);
        RSS = StatisticsHelper<T>.CalculateResidualSumOfSquares(actual, predicted);
        MSE = StatisticsHelper<T>.CalculateMeanSquaredError(actual, predicted);
        RMSE = _numOps.Sqrt(MSE);
        MAPE = StatisticsHelper<T>.CalculateMeanAbsolutePercentageError(actual, predicted);
        MedianAbsoluteError = StatisticsHelper<T>.CalculateMedianAbsoluteError(actual, predicted);
        MaxError = StatisticsHelper<T>.CalculateMaxError(actual, predicted);
        AUCPR = StatisticsHelper<T>.CalculatePrecisionRecallAUC(actual, predicted);
        AUCROC = StatisticsHelper<T>.CalculateROCAUC(actual, predicted);
        SMAPE = StatisticsHelper<T>.CalculateSymmetricMeanAbsolutePercentageError(actual, predicted);
        MeanSquaredLogError = StatisticsHelper<T>.CalculateMeanSquaredLogError(actual, predicted);

        // Calculate standard errors
        SampleStandardError = StatisticsHelper<T>.CalculateSampleStandardError(actual, predicted, numberOfParameters);
        PopulationStandardError = StatisticsHelper<T>.CalculatePopulationStandardError(actual, predicted);

        // Calculate bias and autocorrelation metrics
        MeanBiasError = StatisticsHelper<T>.CalculateMeanBiasError(actual, predicted);
        TheilUStatistic = StatisticsHelper<T>.CalculateTheilUStatistic(actual, predicted);
        DurbinWatsonStatistic = StatisticsHelper<T>.CalculateDurbinWatsonStatistic(actual, predicted);

        // Calculate information criteria
        AIC = StatisticsHelper<T>.CalculateAIC(n, numberOfParameters, RSS);
        BIC = StatisticsHelper<T>.CalculateBIC(n, numberOfParameters, RSS);
        AICAlt = StatisticsHelper<T>.CalculateAICAlternative(n, numberOfParameters, RSS);

        // Calculate classification metrics
        Accuracy = StatisticsHelper<T>.CalculateAccuracy(actual, predicted, predictionType);
        (Precision, Recall, F1Score) = StatisticsHelper<T>.CalculatePrecisionRecallF1(actual, predicted, predictionType);

        // Populate error list
        ErrorList = [..StatisticsHelper<T>.CalculateResiduals(actual, predicted)];
    }

    /// <summary>
    /// Retrieves the value of a specific error metric.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the requested metric.</returns>
    /// <remarks>
    /// <para>
    /// This method allows you to retrieve any of the calculated error metrics by specifying the desired metric type.
    /// It provides a flexible way to access individual metrics without needing to reference specific properties.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like a vending machine for error metrics.
    /// 
    /// You tell it which error metric you want (using the MetricType), and it gives you the value.
    /// For example:
    /// - If you ask for MetricType.MAE, it gives you the Mean Absolute Error
    /// - If you ask for MetricType.RMSE, it gives you the Root Mean Squared Error
    /// 
    /// This is useful when you want to work with different error metrics in a flexible way,
    /// especially if you don't know in advance which metric you'll need.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when an unsupported MetricType is provided.</exception>
    public T GetMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.MAE => MAE,
            MetricType.MSE => MSE,
            MetricType.RMSE => RMSE,
            MetricType.MAPE => MAPE,
            MetricType.MeanBiasError => MeanBiasError,
            MetricType.MedianAbsoluteError => MedianAbsoluteError,
            MetricType.MaxError => MaxError,
            MetricType.TheilUStatistic => TheilUStatistic,
            MetricType.DurbinWatsonStatistic => DurbinWatsonStatistic,
            MetricType.SampleStandardError => SampleStandardError,
            MetricType.PopulationStandardError => PopulationStandardError,
            MetricType.AIC => AIC,
            MetricType.BIC => BIC,
            MetricType.AICAlt => AICAlt,
            MetricType.AUCPR => AUCPR,
            MetricType.AUCROC => AUCROC,
            MetricType.SMAPE => SMAPE,
            MetricType.MeanSquaredLogError => MeanSquaredLogError,
            MetricType.MeanAbsoluteError => MeanAbsoluteError,
            MetricType.MeanSquaredError => MeanSquaredError,
            MetricType.RootMeanSquaredError => RootMeanSquaredError,
            MetricType.Accuracy => Accuracy,
            MetricType.Precision => Precision,
            MetricType.Recall => Recall,
            MetricType.F1Score => F1Score,
            _ => throw new ArgumentException($"Metric {metricType} is not available in ErrorStats.", nameof(metricType)),
        };
    }

    /// <summary>
    /// Checks if a specific metric is available in this ErrorStats instance.
    /// </summary>
    /// <param name="metricType">The type of metric to check for.</param>
    /// <returns>True if the metric is available, false otherwise.</returns>
    /// <remarks>
    /// For Beginners:
    /// This method allows you to check if a particular metric is available before trying to get its value.
    /// It's useful when you're not sure if a specific metric was calculated for this set of errors.
    /// 
    /// For example:
    /// <code>
    /// if (stats.HasMetric(MetricType.MAE))
    /// {
    ///     var maeValue = stats.GetMetric(MetricType.MAE);
    ///     // Use maeValue...
    /// }
    /// </code>
    /// 
    /// This prevents errors that might occur if you try to access a metric that wasn't calculated.
    /// </remarks>
    public bool HasMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.MAE => true,
            MetricType.MSE => true,
            MetricType.RMSE => true,
            MetricType.MAPE => true,
            MetricType.MeanBiasError => true,
            MetricType.MedianAbsoluteError => true,
            MetricType.MaxError => true,
            MetricType.TheilUStatistic => true,
            MetricType.DurbinWatsonStatistic => true,
            MetricType.SampleStandardError => true,
            MetricType.PopulationStandardError => true,
            MetricType.AIC => true,
            MetricType.BIC => true,
            MetricType.AICAlt => true,
            MetricType.AUCPR => true,
            MetricType.AUCROC => true,
            MetricType.SMAPE => true,
            MetricType.MeanSquaredLogError => true,
            MetricType.MeanAbsoluteError => true,
            MetricType.MeanSquaredError => true,
            MetricType.RootMeanSquaredError => true,
            MetricType.Accuracy => true,
            MetricType.Precision => true,
            MetricType.Recall => true,
            MetricType.F1Score => true,
            _ => false,
        };
    }
}