namespace AiDotNet.Models.Results;

/// <summary>
/// Represents the results of a single fold in cross-validation.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., float, double).</typeparam>
/// <typeparam name="TInput">The type of input data (e.g., Matrix&lt;T&gt; for tabular data, Tensor&lt;T&gt; for images).</typeparam>
/// <typeparam name="TOutput">The type of output data (e.g., Vector&lt;T&gt; for predictions, custom types for other formats).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> A FoldResult contains all the performance metrics for one "fold"
/// in cross-validation. Think of it like a report card for a single test of your model,
/// where the model was trained on one subset of your data and tested on another.
/// </para>
/// </remarks>
public class FoldResult<T, TInput, TOutput>
{
    /// <summary>
    /// Gets the index of this fold in the cross-validation process.
    /// </summary>
    public int FoldIndex { get; }

    /// <summary>
    /// Gets the error statistics for the training data.
    /// </summary>
    public ErrorStats<T> TrainingErrors { get; }

    /// <summary>
    /// Gets the error statistics for the validation data.
    /// </summary>
    public ErrorStats<T> ValidationErrors { get; }

    /// <summary>
    /// Gets the prediction statistics for the validation data.
    /// </summary>
    public PredictionStats<T> ValidationPredictionStats { get; }

    /// <summary>
    /// Gets the actual values from the validation dataset.
    /// </summary>
    public Vector<T> ActualValues { get; }

    /// <summary>
    /// Gets the predicted values for the validation dataset.
    /// </summary>
    public Vector<T> PredictedValues { get; }

    /// <summary>
    /// Gets the feature importance scores for this fold.
    /// </summary>
    public Dictionary<string, T> FeatureImportance { get; }

    /// <summary>
    /// Gets the time taken to train the model for this fold.
    /// </summary>
    public TimeSpan TrainingTime { get; }

    /// <summary>
    /// Gets the time taken to evaluate the model for this fold.
    /// </summary>
    public TimeSpan EvaluationTime { get; }

    /// <summary>
    /// Gets the trained model instance for this fold.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property stores the model that was trained specifically on this fold's
    /// training data. Having access to the individual fold models allows you to:
    /// - Analyze how different models vary across folds
    /// - Use ensemble methods by combining predictions from multiple fold models
    /// - Investigate which features are important in different data subsets
    /// </para>
    /// </remarks>
    public IFullModel<T, TInput, TOutput>? Model { get; }

    /// <summary>
    /// Gets the clustering quality metrics for this fold, if applicable.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This property stores clustering quality metrics when your model produces
    /// cluster assignments (like K-Means, DBSCAN, or other clustering algorithms). These metrics help you
    /// understand how well the clustering performed for this specific fold:
    ///
    /// - **Silhouette Score**: How well each point fits in its cluster
    /// - **Calinski-Harabasz Index**: How well-separated and compact clusters are
    /// - **Davies-Bouldin Index**: Average similarity between clusters (lower is better)
    /// - **Adjusted Rand Index**: Similarity to ground truth labels (if available)
    ///
    /// This property will be null if:
    /// - The model doesn't produce cluster labels (e.g., regression or standard classification)
    /// - Clustering metrics couldn't be calculated for this fold
    ///
    /// When cross-validating clustering algorithms, these metrics are automatically calculated
    /// for each fold, allowing you to see how consistent your clustering is across different data splits.
    /// </para>
    /// </remarks>
    public ClusteringMetrics<T>? ClusteringMetrics { get; }

    /// <summary>
    /// Gets the indices of the training samples in this fold.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the row indices from the original dataset that were used
    /// for training the model in this fold. For example, if this contains [0, 1, 5, 6], it means
    /// rows 0, 1, 5, and 6 from the original data were used for training.
    /// </para>
    /// <para>
    /// This is especially useful for:
    /// - Nested cross-validation where you need to extract subsets based on indices
    /// - Debugging to verify correct data splits
    /// - Advanced techniques like stratified sampling verification
    /// </para>
    /// </remarks>
    public int[]? TrainingIndices { get; }

    /// <summary>
    /// Gets the indices of the validation samples in this fold.
    /// </summary>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> These are the row indices from the original dataset that were held out
    /// for validation in this fold. For example, if this contains [2, 3, 4], it means rows 2, 3, and 4
    /// from the original data were used for validation (testing).
    /// </para>
    /// <para>
    /// This ensures you can accurately reconstruct which samples were used for validation, even when
    /// the target values contain duplicates. This prevents data leakage and ensures correct nested
    /// cross-validation.
    /// </para>
    /// </remarks>
    public int[]? ValidationIndices { get; }

    /// <summary>
    /// Creates a new instance of the FoldResult class.
    /// </summary>
    /// <param name="foldIndex">The index of this fold.</param>
    /// <param name="trainingActual">The actual values in the training dataset.</param>
    /// <param name="trainingPredicted">The predicted values for the training dataset.</param>
    /// <param name="validationActual">The actual values in the validation dataset.</param>
    /// <param name="validationPredicted">The predicted values for the validation dataset.</param>
    /// <param name="featureImportance">Optional dictionary of feature importance scores.</param>
    /// <param name="trainingTime">Time taken to train the model.</param>
    /// <param name="evaluationTime">Time taken to evaluate the model.</param>
    /// <param name="featureCount">The number of features used in the model.</param>
    /// <param name="model">Optional trained model instance for this fold.</param>
    /// <param name="clusteringMetrics">Optional clustering quality metrics for this fold.</param>
    /// <param name="trainingIndices">Optional array of indices for training samples in this fold.</param>
    /// <param name="validationIndices">Optional array of indices for validation samples in this fold.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a complete report of how well your model
    /// performed on one fold of cross-validation. It calculates various error metrics and statistics
    /// that help you understand your model's strengths and weaknesses. If your model performs clustering,
    /// you can also include clustering-specific metrics.
    /// </para>
    /// </remarks>
    public FoldResult(
        int foldIndex,
        Vector<T> trainingActual,
        Vector<T> trainingPredicted,
        Vector<T> validationActual,
        Vector<T> validationPredicted,
        Dictionary<string, T>? featureImportance = null,
        TimeSpan? trainingTime = null,
        TimeSpan? evaluationTime = null,
        int featureCount = 0,
        IFullModel<T, TInput, TOutput>? model = null,
        ClusteringMetrics<T>? clusteringMetrics = null,
        int[]? trainingIndices = null,
        int[]? validationIndices = null)
    {
        FoldIndex = foldIndex;
        ActualValues = validationActual;
        PredictedValues = validationPredicted;

        TrainingErrors = new ErrorStats<T>(new ErrorStatsInputs<T>
        {
            Actual = trainingActual,
            Predicted = trainingPredicted,
            FeatureCount = featureCount
        });

        ValidationErrors = new ErrorStats<T>(new ErrorStatsInputs<T>
        {
            Actual = validationActual,
            Predicted = validationPredicted,
            FeatureCount = featureCount
        });

        ValidationPredictionStats = new PredictionStats<T>(new PredictionStatsInputs<T>
        {
            Actual = validationActual,
            Predicted = validationPredicted,
            NumberOfParameters = featureCount,
            ConfidenceLevel = 0.95,
            LearningCurveSteps = 5
        });

        FeatureImportance = featureImportance ?? [];
        TrainingTime = trainingTime ?? TimeSpan.Zero;
        EvaluationTime = evaluationTime ?? TimeSpan.Zero;
        Model = model;
        ClusteringMetrics = clusteringMetrics;
        TrainingIndices = trainingIndices;
        ValidationIndices = validationIndices;
    }
}
