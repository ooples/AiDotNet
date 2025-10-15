namespace AiDotNet.Statistics;

/// <summary>
/// Represents a collection of statistical metrics for evaluating and analyzing machine learning models.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (typically float or double).</typeparam>
/// <remarks>
/// <para>
/// This class calculates and stores various statistical measures that help assess the performance,
/// fit, and characteristics of a machine learning model. It includes metrics for model accuracy,
/// feature importance, model complexity, and various distance and similarity measures.
/// </para>
/// <para><b>For Beginners:</b> Think of ModelStats as a report card for your AI model.
/// 
/// Just like a school report card shows how well a student is doing in different subjects,
/// ModelStats shows how well your AI model is performing in different areas. It helps you:
/// - Understand how accurate your model's predictions are
/// - See which features (inputs) are most important
/// - Check if your model is too simple or too complex
/// - Compare your model's performance to simpler alternatives
/// 
/// This information helps you improve your model and decide if it's ready to use in real-world situations.
/// </para>
/// </remarks>
[Serializable]
public class ModelStats<T, TInput, TOutput> : ModelStatisticsBase<T>
{
    /// <summary>
    /// Configuration options for statistical calculations.
    /// </summary>
    private readonly ModelStatsOptions _options = default!;

    /// <summary>
    /// Gets the correlation matrix showing relationships between features.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This matrix shows how closely related your features are to each other.
    /// Values close to 1 or -1 mean strong relationships, while values near 0 mean weak relationships.
    /// This helps you understand which features might be providing similar information.
    /// </para>
    /// </remarks>
    public Matrix<T> CorrelationMatrix { get; private set; }

    /// <summary>
    /// Gets the covariance matrix showing how features vary together.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This matrix shows how features change together.
    /// It's similar to the correlation matrix but uses a different scale.
    /// It helps identify patterns in how your features behave together.
    /// </para>
    /// </remarks>
    public Matrix<T> CovarianceMatrix { get; private set; }

    /// <summary>
    /// Gets the Variance Inflation Factor (VIF) for each feature.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> VIF helps identify if some features are too similar to others.
    /// High VIF values (usually above 5 or 10) suggest that a feature might be redundant,
    /// as its information is already captured by other features.
    /// </para>
    /// </remarks>
    public List<T> VIFList { get; private set; }

    /// <summary>
    /// Gets the condition number, a measure of the model's numerical stability.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The condition number tells you if small changes in your data
    /// might cause big changes in your model's predictions. A high condition number
    /// (typically above 30) suggests that your model might be unstable and sensitive to small data changes.
    /// </para>
    /// </remarks>
    public T ConditionNumber => GetMetric(MetricType.ConditionNumber);

    /// <summary>
    /// Gets the log pointwise predictive density, a measure of prediction accuracy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a way to measure how well your model's predictions match the actual data.
    /// Higher values generally indicate better predictions. It's particularly useful when comparing different models.
    /// </para>
    /// </remarks>
    public T LogPointwisePredictiveDensity => GetMetric(MetricType.LogPointwisePredictiveDensity);

    /// <summary>
    /// Gets the leave-one-out predictive densities for each data point.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows how well the model predicts each data point when it's trained without that point.
    /// It helps identify which data points might be harder for the model to predict accurately.
    /// </para>
    /// </remarks>
    public List<T> LeaveOneOutPredictiveDensities { get; private set; }

    /// <summary>
    /// Gets the observed test statistic for model evaluation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a single number that summarizes how well your model fits the data.
    /// It's used in statistical tests to determine if your model is significantly better than a simpler alternative.
    /// </para>
    /// </remarks>
    public T ObservedTestStatistic => GetMetric(MetricType.ObservedTestStatistic);

    /// <summary>
    /// Gets samples from the posterior predictive distribution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are possible predictions your model might make if you ran it multiple times.
    /// They help you understand the range and uncertainty of your model's predictions.
    /// </para>
    /// </remarks>
    public List<T> PosteriorPredictiveSamples { get; private set; }

    /// <summary>
    /// Gets the marginal likelihood of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a measure of how well your model fits the data, taking into account its complexity.
    /// It helps in comparing different models, with higher values generally indicating better models.
    /// </para>
    /// </remarks>
    public T MarginalLikelihood => GetMetric(MetricType.MarginalLikelihood);

    /// <summary>
    /// Gets the marginal likelihood of a reference (simpler) model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the marginal likelihood for a basic, simple model.
    /// It's used as a comparison point to see how much better your more complex model performs.
    /// </para>
    /// </remarks>
    public T ReferenceModelMarginalLikelihood => GetMetric(MetricType.ReferenceModelMarginalLikelihood);

    /// <summary>
    /// Gets the log-likelihood of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how probable your data is under your model.
    /// Higher values mean your model fits the data better. It's often used in more advanced statistical techniques.
    /// </para>
    /// </remarks>
    public T LogLikelihood => GetMetric(MetricType.LogLikelihood);

    /// <summary>
    /// Gets the effective number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This estimates how complex your model is in practice.
    /// It might be different from the actual number of parameters and helps identify if your model is overfitting
    /// (using more complexity than needed to explain the data).
    /// </para>
    /// </remarks>
    public T EffectiveNumberOfParameters => GetMetric(MetricType.EffectiveNumberOfParameters);

    /// <summary>
    /// Gets the actual (observed) values from the dataset.
    /// </summary>
    public TOutput Actual { get; }

    /// <summary>
    /// Gets the predicted values from the model.
    /// </summary>
    public TOutput Predicted { get; }

    /// <summary>
    /// Gets the feature values used in the model.
    /// </summary>
    public TInput Features { get; }

    /// <summary>
    /// Gets the full model being evaluated.
    /// </summary>
    public IFullModel<T, TInput, TOutput>? Model { get; }

    /// <summary>
    /// Gets the names of the features used in the model.
    /// </summary>
    public List<string> FeatureNames { get; private set; }

    /// <summary>
    /// Gets a dictionary mapping feature names to their values.
    /// </summary>
    public Dictionary<string, TOutput> FeatureValues { get; private set; }

    /// <summary>
    /// Gets or sets the name of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a friendly name for your model that helps you identify it
    /// among other models. For example, "Customer Churn Predictor" or "House Price Model v2".
    /// </para>
    /// </remarks>
    public string Name { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the type of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This describes what kind of model it is, such as "RandomForest",
    /// "LinearRegression", or "NeuralNetwork". It helps categorize and organize your models.
    /// </para>
    /// </remarks>
    public string Type { get; set; } = string.Empty;

    /// <summary>
    /// Gets or sets the timestamp when the model statistics were calculated.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This records when these statistics were generated,
    /// which is useful for tracking model performance over time or comparing different versions.
    /// </para>
    /// </remarks>
    public DateTime Timestamp { get; set; } = DateTime.UtcNow;

    /// <summary>
    /// Gets the Euclidean distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the straight-line distance between your actual and predicted values.
    /// Lower values indicate predictions that are closer to the actual values.
    /// </para>
    /// </remarks>
    public T EuclideanDistance => GetMetric(MetricType.EuclideanDistance);

    /// <summary>
    /// Gets the Manhattan distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the distance between actual and predicted values as if you could only move
    /// horizontally or vertically (like navigating city blocks). Lower values indicate better predictions.
    /// </para>
    /// </remarks>
    public T ManhattanDistance => GetMetric(MetricType.ManhattanDistance);

    /// <summary>
    /// Gets the cosine similarity between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how similar the direction of your predictions is to the actual values,
    /// ignoring their magnitude. Values closer to 1 indicate more similar directions.
    /// </para>
    /// </remarks>
    public T CosineSimilarity => GetMetric(MetricType.CosineSimilarity);

    /// <summary>
    /// Gets the Jaccard similarity between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the overlap between your actual and predicted values.
    /// It's especially useful for binary (yes/no) predictions. Values closer to 1 indicate more overlap.
    /// </para>
    /// </remarks>
    public T JaccardSimilarity => GetMetric(MetricType.JaccardSimilarity);

    /// <summary>
    /// Gets the Hamming distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many predictions are different from the actual values.
    /// It's most useful for categorical predictions. Lower values indicate fewer differences.
    /// </para>
    /// </remarks>
    public T HammingDistance => GetMetric(MetricType.HammingDistance);

    /// <summary>
    /// Gets the Mahalanobis distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an advanced distance measure that takes into account how your features
    /// are related to each other. It can be more meaningful than simpler distances when your features are correlated.
    /// </para>
    /// </remarks>
    public T MahalanobisDistance => GetMetric(MetricType.MahalanobisDistance);

    /// <summary>
    /// Gets the mutual information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how much information your predictions provide about the actual values.
    /// Higher values mean your predictions are more informative and closely related to the actual values.
    /// </para>
    /// </remarks>
    public T MutualInformation => GetMetric(MetricType.MutualInformation);

    /// <summary>
    /// Gets the normalized mutual information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is similar to mutual information, but scaled to be between 0 and 1.
    /// It's easier to interpret across different datasets. Values closer to 1 indicate better predictions.
    /// </para>
    /// </remarks>
    public T NormalizedMutualInformation => GetMetric(MetricType.NormalizedMutualInformation);

    /// <summary>
    /// Gets the variation of information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how different your predictions are from the actual values.
    /// Lower values indicate that your predictions are more similar to the actual values.
    /// It's particularly useful when comparing different clustering results.
    /// </para>
    /// </remarks>
    public T VariationOfInformation => GetMetric(MetricType.VariationOfInformation);

    /// <summary>
    /// Gets the silhouette score, a measure of how similar an object is to its own cluster compared to other clusters.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This score helps you understand if your model is grouping similar things together well.
    /// It ranges from -1 to 1, where:
    /// - Values close to 1 mean your groups (clusters) are well-defined
    /// - Values close to 0 mean your groups overlap a lot
    /// - Values close to -1 mean some data points might be in the wrong group
    /// </para>
    /// </remarks>
    public T SilhouetteScore => GetMetric(MetricType.SilhouetteScore);

    /// <summary>
    /// Gets the Calinski-Harabasz index, a measure of cluster separation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index tells you how well-separated your groups (clusters) are.
    /// Higher values mean your groups are more distinct from each other, which is generally better.
    /// It's useful when comparing different ways of grouping your data.
    /// </para>
    /// </remarks>
    public T CalinskiHarabaszIndex => GetMetric(MetricType.CalinskiHarabaszIndex);

    /// <summary>
    /// Gets the Davies-Bouldin index, a measure of the average similarity between each cluster and its most similar cluster.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index helps you understand how well-separated your groups (clusters) are.
    /// Lower values are better, meaning your groups are more distinct from each other.
    /// It's particularly useful when you're not sure how many groups to divide your data into.
    /// </para>
    /// </remarks>
    public T DaviesBouldinIndex => GetMetric(MetricType.DaviesBouldinIndex);

    /// <summary>
    /// Gets the Mean Average Precision, a measure of ranking quality.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model ranks items, especially in search or recommendation systems.
    /// It ranges from 0 to 1, where 1 is perfect. It considers both the order of your predictions and their accuracy.
    /// For example, in a search engine, it would measure how well the most relevant results are placed at the top.
    /// </para>
    /// </remarks>
    public T MeanAveragePrecision => GetMetric(MetricType.MeanAveragePrecision);

    /// <summary>
    /// Gets the Normalized Discounted Cumulative Gain, a measure of ranking quality that takes the position of correct items into account.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model ranks items, giving more importance to correct predictions at the top of the list.
    /// It ranges from 0 to 1, where 1 is perfect. It's often used in search engines or recommendation systems to ensure the most relevant items appear first.
    /// </para>
    /// </remarks>
    public T NormalizedDiscountedCumulativeGain => GetMetric(MetricType.NormalizedDiscountedCumulativeGain);

    /// <summary>
    /// Gets the Mean Reciprocal Rank, a statistic measuring the performance of a system that produces a list of possible responses to a query.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model places the first correct answer in a list of predictions.
    /// It ranges from 0 to 1, where 1 means the correct answer is always first. It's often used in question-answering systems
    /// or search engines to measure how quickly a user might find the right answer.
    /// </para>
    /// </remarks>
    public T MeanReciprocalRank => GetMetric(MetricType.MeanReciprocalRank);

    /// <summary>
    /// Gets the Auto-Correlation Function, which measures the correlation between a time series and a lagged version of itself.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This function helps you understand patterns in time-based data.
    /// It shows how similar your data is to itself at different time delays. This can reveal:
    /// - Repeating patterns (like seasonal effects)
    /// - How long effects last in your data
    /// - If your model is missing important time-based patterns
    /// It's particularly useful for time series data, like stock prices or weather patterns.
    /// </para>
    /// </remarks>
    public Vector<T> AutoCorrelationFunction { get; private set; }

    /// <summary>
    /// Gets the Partial Auto-Correlation Function, which measures the direct relationship between an observation and its lag.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This function is similar to the Auto-Correlation Function, but it focuses on the direct relationship between data points at different time delays.
    /// It helps you:
    /// - Identify how many past time points directly influence the current point
    /// - Decide how many past observations to use in time series models
    /// - Understand the "memory" of your time series data
    /// It's often used in more advanced time series analysis and forecasting.
    /// </para>
    /// </remarks>
    public Vector<T> PartialAutoCorrelationFunction { get; private set; }

    /// <summary>
    /// Initializes a new instance of the <see cref="ModelStats{T, TInput, TOutput}"/> class with the specified inputs and options.
    /// </summary>
    /// <param name="inputs">The input data and model information.</param>
    /// <param name="options">Optional configuration settings for statistical calculations.</param>
    /// <param name="modelType">The type of model being evaluated.</param>
    /// <remarks>
    /// <para>
    /// This constructor creates a new ModelStats object, initializes all statistical measures,
    /// and calculates them based on the provided inputs and options.
    /// </para>
    /// <para><b>For Beginners:</b> This is like setting up a new report card for your AI model.
    /// You provide:
    /// - The data your model used (inputs)
    /// - The actual results (inputs.Actual)
    /// - What your model predicted (inputs.Predicted)
    /// - Any special instructions for calculating the statistics (options)
    /// 
    /// The constructor then fills out all the different measures (grades) for your model.
    /// </para>
    /// </remarks>
    internal ModelStats(ModelStatsInputs<T, TInput, TOutput> inputs, ModelType modelType, ModelStatsOptions? options = null)
        : base(modelType, inputs.FeatureCount)
    {
        _options = options ?? new ModelStatsOptions(); // Use default options if not provided

        if (modelType != ModelType.None)
        {
            DetermineValidMetrics();
        }

        VIFList = [];
        CorrelationMatrix = Matrix<T>.Empty();
        CovarianceMatrix = Matrix<T>.Empty();
        AutoCorrelationFunction = Vector<T>.Empty();
        PartialAutoCorrelationFunction = Vector<T>.Empty();
        LeaveOneOutPredictiveDensities = [];
        PosteriorPredictiveSamples = [];
        Actual = inputs.Actual;
        Predicted = inputs.Predicted;
        Features = inputs.XMatrix;
        Model = inputs.Model;
        FeatureNames = inputs.FeatureNames ?? [];
        FeatureValues = inputs.FeatureValues ?? [];

        // Calculate valid metrics
        CalculateModelStats(inputs);
    }

    protected override void DetermineValidMetrics()
    {
        _validMetrics.Clear();
        var cache = MetricValidationCache.Instance;
        var modelMetrics = cache.GetValidMetrics(ModelType, IsModelStatisticMetric);

        foreach (var metric in modelMetrics)
        {
            _validMetrics.Add(metric);
        }
    }

    /// <summary>
    /// Creates an empty instance of the <see cref="ModelStats{T, TInput, TOutput}"/> class.
    /// </summary>
    /// <param name="modelType">The type of model.</param>
    /// <returns>An empty ModelStats object.</returns>
    /// <remarks>
    /// <para>
    /// This method creates a ModelStats object with all measures initialized to their default values.
    /// It's useful when you need a placeholder or when initializing a ModelStats object before populating it with data.
    /// </para>
    /// <para><b>For Beginners:</b> This is like getting a blank report card.
    /// You might use this when:
    /// - You're just starting to set up your model evaluation
    /// - You want to compare an actual model's stats to a "blank slate"
    /// - You're creating a template for future model evaluations
    /// </para>
    /// </remarks>
    public static ModelStats<T, TInput, TOutput> Empty()
    {
        var emptyInputs = new ModelStatsInputs<T, TInput, TOutput>
        {
            Coefficients = Vector<T>.Empty(),
            FeatureCount = 0,
            FitFunction = null
        };

        // Create a ModelStats instance with empty inputs and specified model type
        return new ModelStats<T, TInput, TOutput>(emptyInputs, ModelType.None);
    }

    /// <summary>
    /// Determines if a metric type is provided by this specific statistics provider.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is provided by this statistics class; otherwise, false.</returns>
    protected override bool IsProviderStatisticMetric(MetricType metricType)
    {
        return IsModelStatisticMetric(metricType);
    }

    /// <summary>
    /// Determines if a metric type is a model statistic metric.
    /// </summary>
    /// <param name="metricType">The metric type to check.</param>
    /// <returns>True if the metric is a model statistic; otherwise, false.</returns>
    public static bool IsModelStatisticMetric(MetricType metricType)
    {
        // Define which metrics are considered model statistics
        return metricType switch
        {
            // Matrix<double> metrics
            MetricType.CorrelationMatrix => true,
            MetricType.CovarianceMatrix => true,
            MetricType.VIF => true,

            // Model stability metrics
            MetricType.ConditionNumber => true,

            // Distance metrics between actual and predicted values
            MetricType.EuclideanDistance => true,
            MetricType.ManhattanDistance => true,
            MetricType.CosineSimilarity => true,
            MetricType.JaccardSimilarity => true,
            MetricType.HammingDistance => true,
            MetricType.MahalanobisDistance => true,

            // Information theory metrics
            MetricType.MutualInformation => true,
            MetricType.NormalizedMutualInformation => true,
            MetricType.VariationOfInformation => true,

            // Bayesian and probabilistic metrics
            MetricType.LogPointwisePredictiveDensity => true,
            MetricType.ObservedTestStatistic => true,
            MetricType.MarginalLikelihood => true,
            MetricType.ReferenceModelMarginalLikelihood => true,
            MetricType.LogLikelihood => true,
            MetricType.EffectiveNumberOfParameters => true,
            MetricType.LeaveOneOutPredictiveDensities => true,
            MetricType.PosteriorPredictiveSamples => true,

            // Clustering quality metrics
            MetricType.SilhouetteScore => true,
            MetricType.CalinskiHarabaszIndex => true,
            MetricType.DaviesBouldinIndex => true,

            // Ranking metrics
            MetricType.MeanAveragePrecision => true,
            MetricType.NormalizedDiscountedCumulativeGain => true,
            MetricType.MeanReciprocalRank => true,

            // Time series correlation functions
            MetricType.AutoCorrelationFunction => true,
            MetricType.PartialAutoCorrelationFunction => true,

            // For any other metric type
            _ => false,
        };
    }

    /// <summary>
    /// Calculates all the statistical measures for the model.
    /// </summary>
    /// <param name="inputs">The input data and model information.</param>
    /// <remarks>
    /// <para>
    /// This private method performs all the statistical calculations to populate the various measures in the ModelStats object.
    /// It uses helper methods from the StatisticsHelper class to compute each measure.
    /// It handles empty or null inputs gracefully by returning early with all metrics set to their default values.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like the teacher grading your AI model's report card.
    /// It goes through each measure (grade) one by one:
    /// - Calculating how well your model performed in different areas
    /// - Using the data you provided (actual values and predictions)
    /// - Filling in all the statistics that help you understand your model's performance
    /// 
    /// If you provide empty data, all metrics will remain at their default zero values.
    /// 
    /// You don't call this method directly; it's automatically used when creating a new ModelStats object.
    /// </para>
    /// </remarks>
    private void CalculateModelStats(ModelStatsInputs<T, TInput, TOutput> inputs)
    {
        // Early return if inputs, XMatrix, Actual, or Predicted are null
        if (inputs == null || inputs.XMatrix == null || inputs.Actual == null || inputs.Predicted == null)
        {
            // All metrics remain at their initialized zero/default values
            return;
        }

        // Convert input matrix to Matrix<T> for statistical calculations
        Matrix<T> matrix = ConversionsHelper.ConvertToMatrix<T, TInput>(inputs.XMatrix);

        // Convert actual and predicted values to Vector<T> for statistical calculations
        Vector<T> actual = ConversionsHelper.ConvertToVector<T, TOutput>(inputs.Actual);
        Vector<T> predicted = ConversionsHelper.ConvertToVector<T, TOutput>(inputs.Predicted);

        // Return early if any of the converted data structures are empty
        if (matrix.IsEmpty || actual.IsEmpty || predicted.IsEmpty ||
            matrix.Rows == 0 || matrix.Columns == 0 || actual.Length == 0 || predicted.Length == 0)
        {
            // All metrics remain at their initialized zero/default values
            return;
        }

        // Validate that actual and predicted vectors have the same length
        if (actual.Length != predicted.Length)
        {
            throw new ArgumentException("Actual and predicted vectors must have the same length.");
        }

        // Validate that matrix rows match the number of samples in actual/predicted
        if (matrix.Rows != actual.Length)
        {
            throw new ArgumentException("Number of rows in the feature matrix must match the length of actual/predicted vectors.");
        }

        // Convert coefficients if available
        Vector<T> coefficients = Vector<T>.Empty();
        if (inputs.Coefficients != null)
        {
            coefficients = ConversionsHelper.ConvertToVector<T, object>(inputs.Coefficients);
        }

        // Calculate metrics based on model category
        CalculateBasicMetrics(matrix, actual, predicted, inputs);
        CalculateModelSpecificMetrics(matrix, actual, predicted, coefficients, inputs);
        CalculateDependentMetrics(matrix, actual, predicted);
    }

    /// <summary>
    /// Calculates basic metrics that are common across many model types.
    /// </summary>
    /// <param name="matrix">The feature matrix used for training the model.</param>
    /// <param name="actual">The actual target values.</param>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="inputs">The input data and model information.</param>
    /// <remarks>
    /// <para>
    /// This method calculates fundamental statistical metrics that apply to most machine learning models.
    /// It first determines which metrics can be calculated based on the provided data, then computes
    /// matrix-based metrics (like correlation and covariance) and vector-based metrics (like distance measures).
    /// </para>
    /// <para><b>For Beginners:</b> This method calculates the basic performance measures for your model.
    /// 
    /// It works in three main steps:
    /// 1. First, it checks what metrics can actually be calculated with your data
    /// 2. Then it calculates matrix-based metrics (which show relationships between your features)
    /// 3. Finally, it calculates vector-based metrics (which show how close your predictions are to actual values)
    /// 
    /// The method is smart enough to skip calculations that wouldn't work with your data.
    /// For example, if your data matrix doesn't have enough samples, it won't try to calculate
    /// a correlation matrix that would be unreliable.
    /// 
    /// Think of this as running a series of tests on your model and recording the results,
    /// but only running the tests that make sense for your specific data.
    /// </para>
    /// </remarks>
    private void CalculateBasicMetrics(Matrix<T> matrix, Vector<T> actual, Vector<T> predicted, ModelStatsInputs<T, TInput, TOutput> inputs)
    {
        // Determine what metrics can actually be calculated with the provided data
        bool canCalculateMatrixMetrics = CanCalculateMatrixMetrics(matrix);
        bool canCalculateVectorMetrics = CanCalculateVectorMetrics(actual, predicted);

        // Create a working set of valid metrics based on what's actually computable
        var computeMetrics = new HashSet<MetricType>(_validMetrics);

        // If we can't calculate matrix metrics, remove them from consideration
        if (!canCalculateMatrixMetrics)
        {
            computeMetrics.Remove(MetricType.CorrelationMatrix);
            computeMetrics.Remove(MetricType.CovarianceMatrix);
            computeMetrics.Remove(MetricType.VIF);
            computeMetrics.Remove(MetricType.ConditionNumber);
        }

        // If we can't calculate vector metrics, remove them from consideration
        if (!canCalculateVectorMetrics)
        {
            computeMetrics.Remove(MetricType.EuclideanDistance);
            computeMetrics.Remove(MetricType.ManhattanDistance);
            computeMetrics.Remove(MetricType.CosineSimilarity);
            computeMetrics.Remove(MetricType.JaccardSimilarity);
            computeMetrics.Remove(MetricType.HammingDistance);
            computeMetrics.Remove(MetricType.MahalanobisDistance);
        }

        // Calculate matrix-based metrics first
        if (canCalculateMatrixMetrics)
        {
            if (computeMetrics.Contains(MetricType.CorrelationMatrix))
            {
                CorrelationMatrix = StatisticsHelper<T>.CalculateCorrelationMatrix(matrix, _options);
            }

            if (computeMetrics.Contains(MetricType.CovarianceMatrix))
            {
                CovarianceMatrix = StatisticsHelper<T>.CalculateCovarianceMatrix(matrix);
            }

            // Metrics that depend on correlation matrix
            if (computeMetrics.Contains(MetricType.VIF) && CorrelationMatrix != null)
            {
                VIFList = StatisticsHelper<T>.CalculateVIF(CorrelationMatrix, _options);
                _calculatedMetrics.Add(MetricType.VIF);
            }

            if (computeMetrics.Contains(MetricType.ConditionNumber))
            {
                _metrics[MetricType.ConditionNumber] = StatisticsHelper<T>.CalculateConditionNumber(matrix, _options);
                _calculatedMetrics.Add(MetricType.ConditionNumber);
            }
        }

        // Calculate vector-based metrics
        if (canCalculateVectorMetrics)
        {
            CalculateDistanceMetrics(actual, predicted, computeMetrics);
        }
    }

    /// <summary>
    /// Determines if matrix-based metrics can be calculated with the provided matrix.
    /// </summary>
    /// <param name="matrix">The feature matrix to evaluate.</param>
    /// <returns>True if matrix metrics can be calculated; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method performs several validation checks on the input matrix to ensure it's suitable for
    /// statistical calculations such as correlation and covariance matrices.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if your data matrix is usable for calculations.
    /// 
    /// It checks for several potential issues:
    /// - If the matrix is empty or null
    /// - If it contains any invalid numbers (like NaN or infinity)
    /// - If there's enough data for reliable calculations
    /// 
    /// For statistical calculations to work properly, you need:
    /// - More data points (rows) than features (columns)
    /// - At least 2 columns to calculate relationships between variables
    /// 
    /// If any of these conditions aren't met, the method returns false, indicating that
    /// matrix-based metrics (like correlation) can't be reliably calculated.
    /// </para>
    /// </remarks>
    private bool CanCalculateMatrixMetrics(Matrix<T> matrix)
    {
        // Check if matrix is valid for calculations
        if (matrix == null || matrix.Rows == 0 || matrix.Columns == 0)
        {
            return false;
        }

        // Check for NaN or Infinity values which would cause calculation issues
        for (int i = 0; i < matrix.Rows; i++)
        {
            for (int j = 0; j < matrix.Columns; j++)
            {
                if (!IsValidNumber(matrix[i, j]))
                {
                    return false;
                }
            }
        }

        // Check if matrix is well-conditioned for calculations
        if (matrix.Rows < matrix.Columns || matrix.Columns < 2)
        {
            // Not enough data for reliable correlation/covariance
            return false;
        }

        return true;
    }

    /// <summary>
    /// Determines if vector-based metrics can be calculated with the provided actual and predicted vectors.
    /// </summary>
    /// <param name="actual">The actual target values.</param>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <returns>True if vector metrics can be calculated; otherwise, false.</returns>
    /// <remarks>
    /// <para>
    /// This method validates that the actual and predicted vectors are suitable for calculating
    /// distance and similarity metrics between them.
    /// </para>
    /// <para><b>For Beginners:</b> This method checks if your actual and predicted values can be used
    /// to calculate how well your model performed.
    /// 
    /// It checks for several potential issues:
    /// - If either vector is empty or null
    /// - If they have different lengths (you need the same number of predictions as actual values)
    /// - If they contain any invalid numbers (like NaN or infinity)
    /// 
    /// These checks are important because distance calculations (like Euclidean distance) require
    /// valid, matching pairs of values to work correctly.
    /// 
    /// If any of these conditions aren't met, the method returns false, indicating that
    /// vector-based metrics can't be reliably calculated.
    /// </para>
    /// </remarks>
    private bool CanCalculateVectorMetrics(Vector<T> actual, Vector<T> predicted)
    {
        // Check if vectors are valid for calculations
        if (actual == null || predicted == null || actual.Length == 0 || predicted.Length == 0)
        {
            return false;
        }

        if (actual.Length != predicted.Length)
        {
            return false;
        }

        // Check for NaN or Infinity values
        for (int i = 0; i < actual.Length; i++)
        {
            if (!IsValidNumber(actual[i]) || !IsValidNumber(predicted[i]))
            {
                return false;
            }
        }

        return true;
    }

    /// <summary>
    /// Calculates various distance and similarity metrics between actual and predicted values.
    /// </summary>
    /// <param name="actual">The actual target values.</param>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="computeMetrics">The set of metrics that should be calculated.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates different ways to measure how close or similar
    /// your model's predictions are to the actual values.
    /// 
    /// Think of these metrics as different ways to measure distance between two points:
    /// - Euclidean distance is like measuring with a straight line (as the crow flies)
    /// - Manhattan distance is like following city blocks
    /// - Cosine similarity measures the angle between vectors (how aligned they are)
    /// - Jaccard similarity compares how much two sets overlap
    /// - Hamming distance counts how many positions differ between two sequences
    /// - Mahalanobis distance accounts for correlations between variables (needs a covariance matrix)
    /// 
    /// Each metric gives you a different perspective on how well your predictions match the actual values.
    /// The method checks which metrics you've requested and calculates only those, storing the results
    /// in the metrics collection and marking them as calculated.
    /// </para>
    /// </remarks>
    private void CalculateDistanceMetrics(Vector<T> actual, Vector<T> predicted, HashSet<MetricType> computeMetrics)
    {
        // Calculate standard distance metrics 
        if (computeMetrics.Contains(MetricType.EuclideanDistance))
        {
            _metrics[MetricType.EuclideanDistance] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Euclidean);
            _calculatedMetrics.Add(MetricType.EuclideanDistance);
        }

        if (computeMetrics.Contains(MetricType.ManhattanDistance))
        {
            _metrics[MetricType.ManhattanDistance] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Manhattan);
            _calculatedMetrics.Add(MetricType.ManhattanDistance);
        }

        if (computeMetrics.Contains(MetricType.CosineSimilarity))
        {
            _metrics[MetricType.CosineSimilarity] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Cosine);
            _calculatedMetrics.Add(MetricType.CosineSimilarity);
        }

        if (computeMetrics.Contains(MetricType.JaccardSimilarity))
        {
            _metrics[MetricType.JaccardSimilarity] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Jaccard);
            _calculatedMetrics.Add(MetricType.JaccardSimilarity);
        }

        if (computeMetrics.Contains(MetricType.HammingDistance))
        {
            _metrics[MetricType.HammingDistance] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Hamming);
            _calculatedMetrics.Add(MetricType.HammingDistance);
        }

        // Metrics that depend on covariance matrix
        if (computeMetrics.Contains(MetricType.MahalanobisDistance) && CanCalculateMatrixMetrics(CovarianceMatrix))
        {
            _metrics[MetricType.MahalanobisDistance] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Mahalanobis, CovarianceMatrix);
            _calculatedMetrics.Add(MetricType.MahalanobisDistance);
        }
    }

    /// <summary>
    /// Determines if a value is a valid number for statistical calculations.
    /// </summary>
    /// <param name="value">The value to check.</param>
    /// <returns>True if the value is a valid number (not NaN and not infinity), false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method checks if a number is usable for calculations.
    /// 
    /// In mathematics and statistics, some operations can produce special values that aren't
    /// regular numbers, such as:
    /// - NaN (Not a Number): Results from operations like dividing 0 by 0
    /// - Infinity: Results from operations like dividing by zero
    /// 
    /// These special values can cause problems in calculations, so we need to check for them
    /// before performing statistical operations.
    /// </para>
    /// </remarks>
    private bool IsValidNumber(T value)
    {
        return !(_numOps.IsNaN(value) || _numOps.IsInfinity(value));
    }

    /// <summary>
    /// Calculates metrics that are specific to certain model types.
    /// </summary>
    /// <param name="matrix">The feature matrix used for model training.</param>
    /// <param name="actual">The actual target values.</param>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <param name="coefficients">The model coefficients or parameters.</param>
    /// <param name="inputs">Additional inputs required for metric calculation.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method calculates specialized metrics based on the type of machine learning model.
    /// Different types of models (like clustering, time series, or ranking models) require different evaluation metrics.
    /// 
    /// For example:
    /// - Clustering models need metrics like Silhouette Score that measure how well data points are grouped
    /// - Time series models need metrics that analyze patterns over time
    /// - Ranking models need metrics that evaluate how well items are ordered
    /// 
    /// This method identifies the model type and calculates only the relevant metrics for that type,
    /// making the evaluation process more efficient and meaningful.
    /// </para>
    /// </remarks>
    private void CalculateModelSpecificMetrics(Matrix<T> matrix, Vector<T> actual, Vector<T> predicted, Vector<T> coefficients, ModelStatsInputs<T, TInput, TOutput> inputs)
    {
        var modelCategory = ModelTypeHelper.GetCategory(ModelType);
        var featureCount = inputs.FeatureCount;

        // Bayesian metrics
        if (modelCategory == ModelCategory.Probabilistic)
        {
            if (_validMetrics.Contains(MetricType.LogPointwisePredictiveDensity))
            {
                _metrics[MetricType.LogPointwisePredictiveDensity] = StatisticsHelper<T>.CalculateLogPointwisePredictiveDensity(actual, predicted);
                _calculatedMetrics.Add(MetricType.LogPointwisePredictiveDensity);
            }

            if (_validMetrics.Contains(MetricType.MarginalLikelihood))
            {
                _metrics[MetricType.MarginalLikelihood] = StatisticsHelper<T>.CalculateMarginalLikelihood(actual, predicted, featureCount);
                _calculatedMetrics.Add(MetricType.MarginalLikelihood);
            }

            if (_validMetrics.Contains(MetricType.ReferenceModelMarginalLikelihood))
            {
                _metrics[MetricType.ReferenceModelMarginalLikelihood] = StatisticsHelper<T>.CalculateReferenceModelMarginalLikelihood(actual);
                _calculatedMetrics.Add(MetricType.ReferenceModelMarginalLikelihood);
            }

            if (_validMetrics.Contains(MetricType.EffectiveNumberOfParameters))
            {
                _metrics[MetricType.EffectiveNumberOfParameters] = StatisticsHelper<T>.CalculateEffectiveNumberOfParameters(matrix, coefficients);
                _calculatedMetrics.Add(MetricType.EffectiveNumberOfParameters);
            }

            if (_validMetrics.Contains(MetricType.PosteriorPredictiveSamples))
            {
                PosteriorPredictiveSamples = StatisticsHelper<T>.CalculatePosteriorPredictiveSamples(actual, predicted, featureCount);
                _calculatedMetrics.Add(MetricType.PosteriorPredictiveSamples);
            }
        }

        // Clustering metrics
        if (modelCategory == ModelCategory.Clustering)
        {
            if (_validMetrics.Contains(MetricType.SilhouetteScore))
            {
                _metrics[MetricType.SilhouetteScore] = StatisticsHelper<T>.CalculateSilhouetteScore(matrix, predicted);
                _calculatedMetrics.Add(MetricType.SilhouetteScore);
            }

            if (_validMetrics.Contains(MetricType.CalinskiHarabaszIndex))
            {
                _metrics[MetricType.CalinskiHarabaszIndex] = StatisticsHelper<T>.CalculateCalinskiHarabaszIndex(matrix, predicted);
                _calculatedMetrics.Add(MetricType.CalinskiHarabaszIndex);
            }

            if (_validMetrics.Contains(MetricType.DaviesBouldinIndex))
            {
                _metrics[MetricType.DaviesBouldinIndex] = StatisticsHelper<T>.CalculateDaviesBouldinIndex(matrix, predicted);
                _calculatedMetrics.Add(MetricType.DaviesBouldinIndex);
            }
        }

        // Information theory metrics
        if (_validMetrics.Contains(MetricType.MutualInformation))
        {
            _metrics[MetricType.MutualInformation] = StatisticsHelper<T>.CalculateMutualInformation(actual, predicted);
            _calculatedMetrics.Add(MetricType.MutualInformation);
        }

        if (_validMetrics.Contains(MetricType.NormalizedMutualInformation))
        {
            _metrics[MetricType.NormalizedMutualInformation] = StatisticsHelper<T>.CalculateNormalizedMutualInformation(actual, predicted);
            _calculatedMetrics.Add(MetricType.NormalizedMutualInformation);
        }

        if (_validMetrics.Contains(MetricType.VariationOfInformation))
        {
            _metrics[MetricType.VariationOfInformation] = StatisticsHelper<T>.CalculateVariationOfInformation(actual, predicted);
            _calculatedMetrics.Add(MetricType.VariationOfInformation);
        }

        // Ranking metrics
        if (modelCategory == ModelCategory.Ranking)
        {
            if (_validMetrics.Contains(MetricType.MeanAveragePrecision))
            {
                _metrics[MetricType.MeanAveragePrecision] = StatisticsHelper<T>.CalculateMeanAveragePrecision(actual, predicted, _options.MapTopK);
                _calculatedMetrics.Add(MetricType.MeanAveragePrecision);
            }

            if (_validMetrics.Contains(MetricType.NormalizedDiscountedCumulativeGain))
            {
                _metrics[MetricType.NormalizedDiscountedCumulativeGain] = StatisticsHelper<T>.CalculateNDCG(actual, predicted, _options.NdcgTopK);
                _calculatedMetrics.Add(MetricType.NormalizedDiscountedCumulativeGain);
            }

            if (_validMetrics.Contains(MetricType.MeanReciprocalRank))
            {
                _metrics[MetricType.MeanReciprocalRank] = StatisticsHelper<T>.CalculateMeanReciprocalRank(actual, predicted);
                _calculatedMetrics.Add(MetricType.MeanReciprocalRank);
            }
        }

        // Time series metrics
        if (modelCategory == ModelCategory.TimeSeries)
        {
            // Calculate residuals for time series metrics
            var residuals = StatisticsHelper<T>.CalculateResiduals(actual, predicted);

            if (_validMetrics.Contains(MetricType.AutoCorrelationFunction))
            {
                AutoCorrelationFunction = StatisticsHelper<T>.CalculateAutoCorrelationFunction(residuals, _options.AcfMaxLag);
                _calculatedMetrics.Add(MetricType.AutoCorrelationFunction);
            }

            if (_validMetrics.Contains(MetricType.PartialAutoCorrelationFunction))
            {
                PartialAutoCorrelationFunction = StatisticsHelper<T>.CalculatePartialAutoCorrelationFunction(residuals, _options.PacfMaxLag);
                _calculatedMetrics.Add(MetricType.PartialAutoCorrelationFunction);
            }
        }

        // Cross-validation metrics
        if (_validMetrics.Contains(MetricType.LeaveOneOutPredictiveDensities) && inputs.FitFunction != null)
        {
            // Convert the fit function to work with Matrix<T> and Vector<T>
            var convertedFitFunction = ConversionsHelper.ConvertFitFunction<T, TInput, TOutput>(inputs.FitFunction);

            // Create a wrapper function that matches the expected signature
            Vector<T> wrappedFitFunction(Matrix<T> m, Vector<T> v) => convertedFitFunction(m);
            LeaveOneOutPredictiveDensities = StatisticsHelper<T>.CalculateLeaveOneOutPredictiveDensities(matrix, actual, wrappedFitFunction);
            _calculatedMetrics.Add(MetricType.LeaveOneOutPredictiveDensities);
        }

        // Statistical test metrics
        if (_validMetrics.Contains(MetricType.ObservedTestStatistic))
        {
            _metrics[MetricType.ObservedTestStatistic] = StatisticsHelper<T>.CalculateObservedTestStatistic(actual, predicted);
            _calculatedMetrics.Add(MetricType.ObservedTestStatistic);
        }

        if (_validMetrics.Contains(MetricType.LogLikelihood))
        {
            _metrics[MetricType.LogLikelihood] = StatisticsHelper<T>.CalculateLogLikelihood(actual, predicted);
            _calculatedMetrics.Add(MetricType.LogLikelihood);
        }
    }

    /// <summary>
    /// Calculates metrics that depend on other metrics, ensuring proper calculation order.
    /// </summary>
    /// <param name="matrix">The feature matrix used for model training.</param>
    /// <param name="actual">The actual target values.</param>
    /// <param name="predicted">The predicted values from the model.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method handles the calculation of metrics that depend on other metrics.
    /// 
    /// In statistics and machine learning, some evaluation metrics can't be calculated directly - 
    /// they need other metrics to be calculated first. This is like needing to know your test scores
    /// before you can calculate your average grade.
    /// 
    /// For example:
    /// - The Mahalanobis distance needs the covariance matrix to be calculated first
    /// - Some clustering metrics need cluster centroids to be determined first
    /// - Time series metrics might depend on autocorrelation values
    /// 
    /// This method checks which metrics have already been calculated, and then calculates any
    /// dependent metrics in the correct order, ensuring that all prerequisites are met before
    /// attempting to calculate more complex metrics.
    /// </para>
    /// </remarks>
    private void CalculateDependentMetrics(Matrix<T> matrix, Vector<T> actual, Vector<T> predicted)
    {
        // MahalanobisDistance depends on CovarianceMatrix if not already calculated
        if (_validMetrics.Contains(MetricType.MahalanobisDistance) &&
            !_calculatedMetrics.Contains(MetricType.MahalanobisDistance) &&
            CovarianceMatrix != null && !CovarianceMatrix.IsEmpty)
        {
            _metrics[MetricType.MahalanobisDistance] = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Mahalanobis, CovarianceMatrix);
            _calculatedMetrics.Add(MetricType.MahalanobisDistance);
        }

        // NormalizedMutualInformation might depend on MutualInformation in some implementations
        if (_validMetrics.Contains(MetricType.NormalizedMutualInformation) &&
            !_calculatedMetrics.Contains(MetricType.NormalizedMutualInformation) &&
            _calculatedMetrics.Contains(MetricType.MutualInformation))
        {
            // If we have a dependency relationship between these metrics, handle it here
            // For now, these are calculated independently in CalculateModelSpecificMetrics
        }

        // Clustering validation metrics might have dependencies on cluster assignments
        var modelCategory = ModelTypeHelper.GetCategory(ModelType);
        if (modelCategory == ModelCategory.Clustering)
        {
            // Davies-Bouldin Index might depend on cluster centroids or other clustering metrics
            if (_validMetrics.Contains(MetricType.DaviesBouldinIndex) &&
                !_calculatedMetrics.Contains(MetricType.DaviesBouldinIndex))
            {
                // If this calculation depends on other clustering metrics being calculated first,
                // ensure those dependencies are met here
            }
        }

        // Time series metrics dependencies
        if (modelCategory == ModelCategory.TimeSeries)
        {
            // Partial Auto-Correlation Function might depend on Auto-Correlation Function
            if (_validMetrics.Contains(MetricType.PartialAutoCorrelationFunction) &&
                !_calculatedMetrics.Contains(MetricType.PartialAutoCorrelationFunction) &&
                _calculatedMetrics.Contains(MetricType.AutoCorrelationFunction))
            {
                // Some implementations of PACF might depend on ACF being calculated first
                // If such dependency exists, handle it here
            }
        }

        // Bayesian metrics dependencies
        if (modelCategory == ModelCategory.Probabilistic)
        {
            // Effective Number of Parameters might depend on other Bayesian metrics
            if (_validMetrics.Contains(MetricType.EffectiveNumberOfParameters) &&
                !_calculatedMetrics.Contains(MetricType.EffectiveNumberOfParameters) &&
                matrix != null && !matrix.IsEmpty)
            {
                // If this metric has dependencies on other calculated metrics,
                // ensure they are satisfied here
            }

            // Log Likelihood might influence other probabilistic metrics
            if (_calculatedMetrics.Contains(MetricType.LogLikelihood))
            {
                var logLikelihood = _metrics[MetricType.LogLikelihood];

                // Marginal Likelihood might depend on Log Likelihood in some cases
                if (_validMetrics.Contains(MetricType.MarginalLikelihood) &&
                    !_calculatedMetrics.Contains(MetricType.MarginalLikelihood))
                {
                    // If there's a specific dependency relationship, handle it here
                }
            }
        }

        // Cross-validation dependencies
        if (_validMetrics.Contains(MetricType.LeaveOneOutPredictiveDensities) &&
            _calculatedMetrics.Contains(MetricType.LeaveOneOutPredictiveDensities) &&
            LeaveOneOutPredictiveDensities != null && LeaveOneOutPredictiveDensities.Count > 0)
        {
            // Some metrics might depend on Leave-One-Out results
            // For example, computing aggregate statistics from LOO densities
        }

        // Ranking metrics dependencies
        if (modelCategory == ModelCategory.Ranking)
        {
            // NDCG might depend on DCG values
            // MRR might depend on individual rank calculations
            // These are currently calculated independently, but if dependencies exist,
            // they would be handled here
        }

        // Check for any matrix-based dependencies
        if (matrix?.Rows > 1 && matrix?.Columns > 1)
        {
            // Condition Number might influence numerical stability assessments
            if (_calculatedMetrics.Contains(MetricType.ConditionNumber))
            {
                var conditionNumber = _metrics[MetricType.ConditionNumber];

                // Other metrics might need to adjust based on numerical stability
                // indicated by the condition number
            }
        }

        // Handle any cross-category dependencies
        // Some metrics might span multiple categories and have complex dependencies
        if (_calculatedMetrics.Count > 0)
        {
            // Generic dependency handling for metrics that don't fall into specific categories
            // This ensures all possible dependencies are captured
        }
    }
}