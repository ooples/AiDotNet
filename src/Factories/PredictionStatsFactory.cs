namespace AiDotNet.Factories;

/// <summary>
/// Factory class for creating PredictionStats instances.
/// </summary>
/// <remarks>
/// <para>
/// This static class provides factory methods to create PredictionStats instances with
/// different options and parameters. It simplifies the creation of PredictionStats objects
/// by providing convenient overloads and sensible defaults.
/// </para>
/// <para>
/// <b>For Beginners:</b> A factory class provides convenient methods for creating
/// objects. Rather than directly calling the constructor with many parameters,
/// you can use these methods which handle a lot of the details for you.
/// </para>
/// </remarks>
public static class PredictionStatsFactory
{
    /// <summary>
    /// Creates a PredictionStats instance for a specific model type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">Vector<double> of actual values (ground truth).</param>
    /// <param name="predicted">Vector<double> of predicted values from your model.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="numberOfParameters">Number of features or parameters in your model.</param>
    /// <param name="confidenceLevel">The confidence level for statistical intervals (e.g., 0.95 for 95% confidence).</param>
    /// <param name="learningCurveSteps">Number of steps to use when calculating the learning curve.</param>
    /// <param name="predictionType">The type of prediction (regression, binary classification, etc.).</param>
    /// <returns>A PredictionStats instance with calculated metrics appropriate for the model type.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a PredictionStats object with the specified parameters,
    /// calculates all appropriate metrics and intervals, and returns the instance.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This is a convenience method for creating a PredictionStats object in one step.
    /// You provide your actual and predicted values, the type of model you're evaluating,
    /// and additional parameters, then it creates a PredictionStats object with all the
    /// appropriate prediction metrics calculated.
    /// </para>
    /// </remarks>
    public static PredictionStats<T> Create<T>(
        Vector<T> actual,
        Vector<T> predicted,
        ModelType modelType,
        int numberOfParameters = 1,
        double confidenceLevel = 0.95,
        int learningCurveSteps = 5,
        PredictionType predictionType = PredictionType.Regression)
    {
        var inputs = new PredictionStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            NumberOfParameters = numberOfParameters,
            ConfidenceLevel = confidenceLevel,
            LearningCurveSteps = learningCurveSteps,
            PredictionType = predictionType
        };

        return new PredictionStats<T>(inputs, modelType);
    }

    /// <summary>
    /// Creates a PredictionStats instance for a specific model type with progress reporting and cancellation support.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">Vector<double> of actual values (ground truth).</param>
    /// <param name="predicted">Vector<double> of predicted values from your model.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="numberOfParameters">Number of features or parameters in your model.</param>
    /// <param name="confidenceLevel">The confidence level for statistical intervals (e.g., 0.95 for 95% confidence).</param>
    /// <param name="learningCurveSteps">Number of steps to use when calculating the learning curve.</param>
    /// <param name="predictionType">The type of prediction (regression, binary classification, etc.).</param>
    /// <param name="progress">Optional progress reporting.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>A PredictionStats instance with calculated metrics appropriate for the model type.</returns>
    /// <remarks>
    /// <para>
    /// This method is an extension of the Create method that adds support for
    /// progress reporting and cancellation. It's useful for long-running calculations
    /// that might need to be monitored or canceled.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> This version of the Create method includes support for progress reporting and cancellation.
    /// Progress reporting lets you track how far along the calculations are, which is useful for
    /// showing progress bars or status updates to users. Cancellation allows you to stop the
    /// calculations early if needed, which is useful for responsive user interfaces.
    /// </para>
    /// </remarks>
    public static PredictionStats<T> Create<T>(
        Vector<T> actual,
        Vector<T> predicted,
        ModelType modelType,
        int numberOfParameters = 1,
        double confidenceLevel = 0.95,
        int learningCurveSteps = 5,
        PredictionType predictionType = PredictionType.Regression,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var inputs = new PredictionStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            NumberOfParameters = numberOfParameters,
            ConfidenceLevel = confidenceLevel,
            LearningCurveSteps = learningCurveSteps,
            PredictionType = predictionType
        };

        return new PredictionStats<T>(inputs, modelType, progress, cancellationToken);
    }

    /// <summary>
    /// Creates a PredictionStats instance specifically for time series forecasting models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">Vector<double> of actual values (ground truth).</param>
    /// <param name="predicted">Vector<double> of predicted values from your model.</param>
    /// <param name="modelType">The specific time series model type being evaluated.</param>
    /// <param name="numberOfParameters">Number of features or parameters in your model.</param>
    /// <param name="confidenceLevel">The confidence level for statistical intervals (e.g., 0.95 for 95% confidence).</param>
    /// <param name="forecastHorizon">The number of time steps ahead that are being forecast.</param>
    /// <returns>A PredictionStats instance with time series-specific metrics calculated.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a PredictionStats object specifically tailored for time series models.
    /// It ensures that appropriate time series metrics and intervals are calculated.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this method when you're evaluating forecasting models
    /// that predict values over time. It includes special metrics and intervals
    /// that are specifically designed for time series data.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when a non-time series model type is provided.</exception>
    public static PredictionStats<T> CreateForTimeSeries<T>(
        Vector<T> actual,
        Vector<T> predicted,
        ModelType modelType,
        int numberOfParameters = 1,
        double confidenceLevel = 0.95,
        int forecastHorizon = 1)
    {
        // Verify that the model type is a time series model
        if (ModelTypeHelper.GetCategory(modelType) != ModelCategory.TimeSeries)
        {
            throw new ArgumentException($"Model type {modelType} is not a time series model. Use a time series model type or the regular Create method instead.");
        }

        var inputs = new PredictionStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            NumberOfParameters = numberOfParameters,
            ConfidenceLevel = confidenceLevel,
            LearningCurveSteps = 0, // Learning curves aren't typically used for time series
            PredictionType = PredictionType.Regression, // Time series forecasting is typically regression
            ForecastHorizon = forecastHorizon
        };

        return new PredictionStats<T>(inputs, modelType);
    }

    /// <summary>
    /// Creates a PredictionStats instance specifically for classification models.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">Vector<double> of actual class labels.</param>
    /// <param name="predicted">Vector<double> of predicted class probabilities or labels.</param>
    /// <param name="modelType">The specific classification model type being evaluated.</param>
    /// <param name="numberOfParameters">Number of features or parameters in your model.</param>
    /// <param name="confidenceLevel">The confidence level for statistical intervals (e.g., 0.95 for 95% confidence).</param>
    /// <param name="numberOfClasses">The number of classes in the classification problem.</param>
    /// <param name="isBinary">Whether this is a binary classification problem.</param>
    /// <returns>A PredictionStats instance with classification-specific metrics calculated.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a PredictionStats object specifically tailored for classification models.
    /// It ensures that appropriate classification metrics like accuracy, precision, recall, and F1 score are calculated.
    /// </para>
    /// <para>
    /// <b>For Beginners:</b> Use this method when you're evaluating models that classify data into
    /// categories (like spam/not spam, or dog/cat/bird). It includes metrics that are designed
    /// specifically for measuring classification performance.
    /// </para>
    /// </remarks>
    public static PredictionStats<T> CreateForClassification<T>(
        Vector<T> actual,
        Vector<T> predicted,
        ModelType modelType,
        int numberOfParameters = 1,
        double confidenceLevel = 0.95,
        int numberOfClasses = 2,
        bool isBinary = true)
    {
        var inputs = new PredictionStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            NumberOfParameters = numberOfParameters,
            ConfidenceLevel = confidenceLevel,
            LearningCurveSteps = 5,
            PredictionType = isBinary ? PredictionType.BinaryClassification : PredictionType.MulticlassClassification,
            NumberOfClasses = numberOfClasses
        };

        return new PredictionStats<T>(inputs, modelType);
    }
}