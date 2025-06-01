namespace AiDotNet.Factories;

/// <summary>
/// Factory class for creating ErrorStats instances.
/// </summary>
public static class ErrorStatsFactory
{
    /// <summary>
    /// Creates an ErrorStats instance for a specific model type.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">Vector<double> of actual values (ground truth).</param>
    /// <param name="predicted">Vector<double> of predicted values from your model.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="taskType">Optional task type for neural networks.</param>
    /// <param name="progress">Optional progress reporting.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <param name="featureCount">Number of features or parameters in your model.</param>
    /// <returns>An ErrorStats instance with calculated metrics appropriate for the model type.</returns>
    /// <remarks>
    /// For Beginners:
    /// This is a convenience method for creating an ErrorStats object in one step.
    /// You provide your actual and predicted values, the type of model you're evaluating,
    /// and optionally the number of features in your model. It then creates an ErrorStats
    /// object with all the appropriate error metrics calculated.
    /// </remarks>
    public static ErrorStats<T> Create<T>(Vector<T> actual, Vector<T> predicted, ModelType modelType, NeuralNetworkTaskType? taskType = null, IProgress<double>? progress = null, CancellationToken cancellationToken = default, int featureCount = 1)
    {
        var inputs = new ErrorStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = featureCount
        };

        return new ErrorStats<T>(inputs, modelType, taskType, progress, cancellationToken);
    }

    /// <summary>
    /// Creates an ErrorStats instance for a specific model type with progress reporting and cancellation support.
    /// </summary>
    /// <typeparam name="T">The numeric type used for calculations.</typeparam>
    /// <param name="actual">Vector<double> of actual values (ground truth).</param>
    /// <param name="predicted">Vector<double> of predicted values from your model.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <param name="featureCount">Number of features or parameters in your model.</param>
    /// <param name="progress">Optional progress reporting.</param>
    /// <param name="cancellationToken">Optional cancellation token.</param>
    /// <returns>An ErrorStats instance with calculated metrics appropriate for the model type.</returns>
    /// <remarks>
    /// For Beginners:
    /// This version of the Create method includes support for progress reporting and cancellation.
    /// Progress reporting lets you track how far along the calculations are, which is useful for
    /// showing progress bars or status updates to users. Cancellation allows you to stop the
    /// calculations early if needed, which is useful for responsive user interfaces.
    /// </remarks>
    public static ErrorStats<T> Create<T>(
        Vector<T> actual,
        Vector<T> predicted,
        ModelType modelType,
        int featureCount = 1,
        NeuralNetworkTaskType? taskType = null,
        IProgress<double>? progress = null,
        CancellationToken cancellationToken = default)
    {
        var inputs = new ErrorStatsInputs<T>
        {
            Actual = actual,
            Predicted = predicted,
            FeatureCount = featureCount
        };

        return new ErrorStats<T>(inputs, modelType, taskType, progress, cancellationToken);
    }
}