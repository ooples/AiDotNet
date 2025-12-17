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
public class ModelStats<T, TInput, TOutput>
{
    private readonly INumericOperations<T> _numOps;
    private readonly ModelStatsOptions _options;

    /// <summary>
    /// Gets the number of features (input variables) used in the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the number of different pieces of information your model uses to make predictions.
    /// For example, if you're predicting house prices, features might include size, number of bedrooms, location, etc.
    /// </para>
    /// </remarks>
    public int FeatureCount { get; private set; }

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
    public T ConditionNumber { get; private set; }

    /// <summary>
    /// Gets the log pointwise predictive density, a measure of prediction accuracy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a way to measure how well your model's predictions match the actual data.
    /// Higher values generally indicate better predictions. It's particularly useful when comparing different models.
    /// </para>
    /// </remarks>
    public T LogPointwisePredictiveDensity { get; private set; }

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
    public T ObservedTestStatistic { get; private set; }

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
    public T MarginalLikelihood { get; private set; }

    /// <summary>
    /// Gets the marginal likelihood of a reference (simpler) model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the marginal likelihood for a basic, simple model.
    /// It's used as a comparison point to see how much better your more complex model performs.
    /// </para>
    /// </remarks>
    public T ReferenceModelMarginalLikelihood { get; private set; }

    /// <summary>
    /// Gets the log-likelihood of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how probable your data is under your model.
    /// Higher values mean your model fits the data better. It's often used in more advanced statistical techniques.
    /// </para>
    /// </remarks>
    public T LogLikelihood { get; private set; }

    /// <summary>
    /// Gets the effective number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This estimates how complex your model is in practice.
    /// It might be different from the actual number of parameters and helps identify if your model is overfitting
    /// (using more complexity than needed to explain the data).
    /// </para>
    /// </remarks>
    public T EffectiveNumberOfParameters { get; private set; }

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
    /// Gets the Euclidean distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the straight-line distance between your actual and predicted values.
    /// Lower values indicate predictions that are closer to the actual values.
    /// </para>
    /// </remarks>
    public T EuclideanDistance { get; private set; }

    /// <summary>
    /// Gets the Manhattan distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the distance between actual and predicted values as if you could only move
    /// horizontally or vertically (like navigating city blocks). Lower values indicate better predictions.
    /// </para>
    /// </remarks>
    public T ManhattanDistance { get; private set; }

    /// <summary>
    /// Gets the cosine similarity between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how similar the direction of your predictions is to the actual values,
    /// ignoring their magnitude. Values closer to 1 indicate more similar directions.
    /// </para>
    /// </remarks>
    public T CosineSimilarity { get; private set; }

    /// <summary>
    /// Gets the Jaccard similarity between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the overlap between your actual and predicted values.
    /// It's especially useful for binary (yes/no) predictions. Values closer to 1 indicate more overlap.
    /// </para>
    /// </remarks>
    public T JaccardSimilarity { get; private set; }

    /// <summary>
    /// Gets the Hamming distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many predictions are different from the actual values.
    /// It's most useful for categorical predictions. Lower values indicate fewer differences.
    /// </para>
    /// </remarks>
    public T HammingDistance { get; private set; }

    /// <summary>
    /// Gets the Mahalanobis distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an advanced distance measure that takes into account how your features
    /// are related to each other. It can be more meaningful than simpler distances when your features are correlated.
    /// </para>
    /// </remarks>
    public T MahalanobisDistance { get; private set; }

    /// <summary>
    /// Gets the mutual information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how much information your predictions provide about the actual values.
    /// Higher values mean your predictions are more informative and closely related to the actual values.
    /// </para>
    /// </remarks>
    public T MutualInformation { get; private set; }

    /// <summary>
    /// Gets the normalized mutual information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is similar to mutual information, but scaled to be between 0 and 1.
    /// It's easier to interpret across different datasets. Values closer to 1 indicate better predictions.
    /// </para>
    /// </remarks>
    public T NormalizedMutualInformation { get; private set; }

    /// <summary>
    /// Gets the variation of information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how different your predictions are from the actual values.
    /// Lower values indicate that your predictions are more similar to the actual values.
    /// It's particularly useful when comparing different clustering results.
    /// </para>
    /// </remarks>
    public T VariationOfInformation { get; private set; }

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
    public T SilhouetteScore { get; private set; }

    /// <summary>
    /// Gets the Calinski-Harabasz index, a measure of cluster separation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index tells you how well-separated your groups (clusters) are.
    /// Higher values mean your groups are more distinct from each other, which is generally better.
    /// It's useful when comparing different ways of grouping your data.
    /// </para>
    /// </remarks>
    public T CalinskiHarabaszIndex { get; private set; }

    /// <summary>
    /// Gets the Davies-Bouldin index, a measure of the average similarity between each cluster and its most similar cluster.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index helps you understand how well-separated your groups (clusters) are.
    /// Lower values are better, meaning your groups are more distinct from each other.
    /// It's particularly useful when you're not sure how many groups to divide your data into.
    /// </para>
    /// </remarks>
    public T DaviesBouldinIndex { get; private set; }

    /// <summary>
    /// Gets the Mean Average Precision, a measure of ranking quality.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model ranks items, especially in search or recommendation systems.
    /// It ranges from 0 to 1, where 1 is perfect. It considers both the order of your predictions and their accuracy.
    /// For example, in a search engine, it would measure how well the most relevant results are placed at the top.
    /// </para>
    /// </remarks>
    public T MeanAveragePrecision { get; private set; }

    /// <summary>
    /// Gets the Normalized Discounted Cumulative Gain, a measure of ranking quality that takes the position of correct items into account.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model ranks items, giving more importance to correct predictions at the top of the list.
    /// It ranges from 0 to 1, where 1 is perfect. It's often used in search engines or recommendation systems to ensure the most relevant items appear first.
    /// </para>
    /// </remarks>
    public T NormalizedDiscountedCumulativeGain { get; private set; }

    /// <summary>
    /// Gets the Mean Reciprocal Rank, a statistic measuring the performance of a system that produces a list of possible responses to a query.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model places the first correct answer in a list of predictions.
    /// It ranges from 0 to 1, where 1 means the correct answer is always first. It's often used in question-answering systems
    /// or search engines to measure how quickly a user might find the right answer.
    /// </para>
    /// </remarks>
    public T MeanReciprocalRank { get; private set; }

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
    /// Initializes a new instance of the <see cref="ModelStats{T}"/> class with the specified inputs and options.
    /// </summary>
    /// <param name="inputs">The input data and model information.</param>
    /// <param name="options">Optional configuration settings for statistical calculations.</param>
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
    internal ModelStats(ModelStatsInputs<T, TInput, TOutput> inputs, ModelStatsOptions? options = null)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _options = options ?? new ModelStatsOptions(); // Use default options if not provided
        FeatureCount = inputs.FeatureCount;
        ConditionNumber = _numOps.Zero;
        VIFList = [];
        CorrelationMatrix = Matrix<T>.Empty();
        CovarianceMatrix = Matrix<T>.Empty();
        AutoCorrelationFunction = Vector<T>.Empty();
        PartialAutoCorrelationFunction = Vector<T>.Empty();
        MeanAveragePrecision = _numOps.Zero;
        NormalizedDiscountedCumulativeGain = _numOps.Zero;
        MeanReciprocalRank = _numOps.Zero;
        SilhouetteScore = _numOps.Zero;
        CalinskiHarabaszIndex = _numOps.Zero;
        DaviesBouldinIndex = _numOps.Zero;
        MutualInformation = _numOps.Zero;
        NormalizedMutualInformation = _numOps.Zero;
        VariationOfInformation = _numOps.Zero;
        EuclideanDistance = _numOps.Zero;
        ManhattanDistance = _numOps.Zero;
        CosineSimilarity = _numOps.Zero;
        JaccardSimilarity = _numOps.Zero;
        HammingDistance = _numOps.Zero;
        MahalanobisDistance = _numOps.Zero;
        LogPointwisePredictiveDensity = _numOps.Zero;
        LeaveOneOutPredictiveDensities = [];
        ObservedTestStatistic = _numOps.Zero;
        PosteriorPredictiveSamples = [];
        MarginalLikelihood = _numOps.Zero;
        ReferenceModelMarginalLikelihood = _numOps.Zero;
        LogLikelihood = _numOps.Zero;
        EffectiveNumberOfParameters = _numOps.Zero;
        Actual = inputs.Actual;
        Predicted = inputs.Predicted;
        Features = inputs.XMatrix;
        Model = inputs.Model;
        FeatureNames = inputs.FeatureNames ?? [];
        FeatureValues = inputs.FeatureValues ?? [];

        // Only calculate statistics if we have actual data to analyze
        // Skip calculation for empty ModelStats (placeholder objects)
        if (!IsEmptyInput(inputs))
        {
            CalculateModelStats(inputs);
        }
    }

    /// <summary>
    /// Determines whether the input data is empty or uninitialized.
    /// </summary>
    /// <param name="inputs">The input data to check.</param>
    /// <returns>True if the inputs are empty; otherwise, false.</returns>
    private static bool IsEmptyInput(ModelStatsInputs<T, TInput, TOutput> inputs)
    {
        // Check if the XMatrix is empty based on its type
        if (inputs.XMatrix is Matrix<T> matrix)
        {
            return matrix.Rows == 0 || matrix.Columns == 0;
        }
        else if (inputs.XMatrix is Tensor<T> tensor)
        {
            return tensor.Rank == 0 || tensor.Length == 0;
        }

        // For unknown types, assume not empty to maintain existing behavior
        return false;
    }

    /// <summary>
    /// Creates an empty instance of the <see cref="ModelStats{T}"/> class.
    /// </summary>
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
        return new ModelStats<T, TInput, TOutput>(new());
    }

    /// <summary>
    /// Calculates all the statistical measures for the model.
    /// </summary>
    /// <param name="inputs">The input data and model information.</param>
    /// <remarks>
    /// <para>
    /// This private method performs all the statistical calculations to populate the various measures in the ModelStats object.
    /// It uses helper methods from the StatisticsHelper class to compute each measure.
    /// </para>
    /// <para><b>For Beginners:</b> This method is like the teacher grading your AI model's report card.
    /// It goes through each measure (grade) one by one:
    /// - Calculating how well your model performed in different areas
    /// - Using the data you provided (actual values and predictions)
    /// - Filling in all the statistics that help you understand your model's performance
    /// 
    /// You don't call this method directly; it's automatically used when creating a new ModelStats object.
    /// </para>
    /// </remarks>
    private void CalculateModelStats(ModelStatsInputs<T, TInput, TOutput> inputs)
    {
        // Convert input matrix to Matrix<T> for statistical calculations
        Matrix<T> matrix = ConversionsHelper.ConvertToMatrix<T, TInput>(inputs.XMatrix);

        // Convert actual and predicted values to Vector<T> for statistical calculations
        Vector<T> actual = ConversionsHelper.ConvertToVector<T, TOutput>(inputs.Actual);
        Vector<T> predicted = ConversionsHelper.ConvertToVector<T, TOutput>(inputs.Predicted);

        // Convert coefficients if available
        Vector<T> coefficients = Vector<T>.Empty();
        if (inputs.Coefficients != null)
        {
            coefficients = ConversionsHelper.ConvertToVector<T, object>(inputs.Coefficients);
        }

        var featureCount = inputs.FeatureCount;

        // Calculate all statistical metrics using the converted data types
        CorrelationMatrix = StatisticsHelper<T>.CalculateCorrelationMatrix(matrix, _options);
        CovarianceMatrix = StatisticsHelper<T>.CalculateCovarianceMatrix(matrix);
        VIFList = StatisticsHelper<T>.CalculateVIF(CorrelationMatrix, _options);
        ConditionNumber = StatisticsHelper<T>.CalculateConditionNumber(matrix, _options);
        LogPointwisePredictiveDensity = StatisticsHelper<T>.CalculateLogPointwisePredictiveDensity(actual, predicted);

        if (inputs.FitFunction != null)
        {
            // Convert the fit function to work with Matrix<T> and Vector<T>
            var convertedFitFunction = ConversionsHelper.ConvertFitFunction<T, TInput, TOutput>(inputs.FitFunction);

            // Create a wrapper function that matches the expected signature
            Vector<T> wrappedFitFunction(Matrix<T> m, Vector<T> v) => convertedFitFunction(m);
            LeaveOneOutPredictiveDensities = StatisticsHelper<T>.CalculateLeaveOneOutPredictiveDensities(matrix, actual, wrappedFitFunction);
        }

        ObservedTestStatistic = StatisticsHelper<T>.CalculateObservedTestStatistic(actual, predicted);
        PosteriorPredictiveSamples = StatisticsHelper<T>.CalculatePosteriorPredictiveSamples(actual, predicted, featureCount);
        MarginalLikelihood = StatisticsHelper<T>.CalculateMarginalLikelihood(actual, predicted, featureCount);
        ReferenceModelMarginalLikelihood = StatisticsHelper<T>.CalculateReferenceModelMarginalLikelihood(actual);
        LogLikelihood = StatisticsHelper<T>.CalculateLogLikelihood(actual, predicted);
        EffectiveNumberOfParameters = StatisticsHelper<T>.CalculateEffectiveNumberOfParameters(matrix, coefficients);
        MutualInformation = StatisticsHelper<T>.CalculateMutualInformation(actual, predicted);
        NormalizedMutualInformation = StatisticsHelper<T>.CalculateNormalizedMutualInformation(actual, predicted);
        VariationOfInformation = StatisticsHelper<T>.CalculateVariationOfInformation(actual, predicted);
        SilhouetteScore = StatisticsHelper<T>.CalculateSilhouetteScore(matrix, predicted);
        CalinskiHarabaszIndex = StatisticsHelper<T>.CalculateCalinskiHarabaszIndex(matrix, predicted);
        DaviesBouldinIndex = StatisticsHelper<T>.CalculateDaviesBouldinIndex(matrix, predicted);
        MeanAveragePrecision = StatisticsHelper<T>.CalculateMeanAveragePrecision(actual, predicted, _options.MapTopK);
        NormalizedDiscountedCumulativeGain = StatisticsHelper<T>.CalculateNDCG(actual, predicted, _options.NdcgTopK);
        MeanReciprocalRank = StatisticsHelper<T>.CalculateMeanReciprocalRank(actual, predicted);

        // Calculate residuals and time series metrics
        var residuals = StatisticsHelper<T>.CalculateResiduals(actual, predicted);
        AutoCorrelationFunction = StatisticsHelper<T>.CalculateAutoCorrelationFunction(residuals, _options.AcfMaxLag);
        PartialAutoCorrelationFunction = StatisticsHelper<T>.CalculatePartialAutoCorrelationFunction(residuals, _options.PacfMaxLag);

        // Calculate distance metrics
        EuclideanDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Euclidean);
        ManhattanDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Manhattan);
        CosineSimilarity = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Cosine);
        JaccardSimilarity = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Jaccard);
        HammingDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Hamming);

        // Mahalanobis distance requires vector dimensions to match covariance matrix dimensions
        // Skip calculation if dimensions don't match (covariance is feature x feature, not sample x sample)
        try
        {
            if (CovarianceMatrix.Rows == actual.Length && CovarianceMatrix.Columns == actual.Length)
            {
                MahalanobisDistance = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Mahalanobis, CovarianceMatrix);
            }
        }
        catch (ArgumentException)
        {
            // Silently skip Mahalanobis distance calculation when dimensions don't match
        }
    }

    /// <summary>
    /// Retrieves the value of a specific metric.
    /// </summary>
    /// <param name="metricType">The type of metric to retrieve.</param>
    /// <returns>The value of the specified metric.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method allows you to get the value of any metric calculated by ModelStats.
    /// You specify which metric you want using the MetricType enum, and the method returns its value.
    /// </para>
    /// <para>
    /// For example, if you want to get the Euclidean Distance, you would call:
    /// <code>
    /// T euclideanDistance = modelStats.GetMetric(MetricType.EuclideanDistance);
    /// </code>
    /// </para>
    /// <para>
    /// This is useful when you want to programmatically access different metrics without
    /// needing to know the specific property names for each one.
    /// </para>
    /// </remarks>
    /// <exception cref="ArgumentException">Thrown when an unsupported metric type is requested.</exception>
    public T GetMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.ConditionNumber => ConditionNumber,
            MetricType.LogPointwisePredictiveDensity => LogPointwisePredictiveDensity,
            MetricType.ObservedTestStatistic => ObservedTestStatistic,
            MetricType.MarginalLikelihood => MarginalLikelihood,
            MetricType.ReferenceModelMarginalLikelihood => ReferenceModelMarginalLikelihood,
            MetricType.LogLikelihood => LogLikelihood,
            MetricType.EffectiveNumberOfParameters => EffectiveNumberOfParameters,
            MetricType.EuclideanDistance => EuclideanDistance,
            MetricType.ManhattanDistance => ManhattanDistance,
            MetricType.CosineSimilarity => CosineSimilarity,
            MetricType.JaccardSimilarity => JaccardSimilarity,
            MetricType.HammingDistance => HammingDistance,
            MetricType.MahalanobisDistance => MahalanobisDistance,
            MetricType.MutualInformation => MutualInformation,
            MetricType.NormalizedMutualInformation => NormalizedMutualInformation,
            MetricType.VariationOfInformation => VariationOfInformation,
            MetricType.SilhouetteScore => SilhouetteScore,
            MetricType.CalinskiHarabaszIndex => CalinskiHarabaszIndex,
            MetricType.DaviesBouldinIndex => DaviesBouldinIndex,
            MetricType.MeanAveragePrecision => MeanAveragePrecision,
            MetricType.NormalizedDiscountedCumulativeGain => NormalizedDiscountedCumulativeGain,
            MetricType.MeanReciprocalRank => MeanReciprocalRank,
            _ => throw new ArgumentException($"Metric {metricType} is not available in ModelStats.", nameof(metricType)),
        };
    }

    /// <summary>
    /// Checks if a specific metric is available in this ModelStats instance.
    /// </summary>
    /// <param name="metricType">The type of metric to check for.</param>
    /// <returns>True if the metric is available, false otherwise.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method lets you check if a particular metric has been calculated 
    /// for your model. It's useful when you're not sure if a specific metric is available, 
    /// especially when working with different types of models or datasets.
    /// </para>
    /// <para>
    /// For example, if you want to check if the Euclidean Distance is available, you would call:
    /// <code>
    /// if (modelStats.HasMetric(MetricType.EuclideanDistance))
    /// {
    ///     var distance = modelStats.GetMetric(MetricType.EuclideanDistance);
    ///     // Use the distance...
    /// }
    /// </code>
    /// </para>
    /// <para>
    /// This prevents errors that might occur if you try to access a metric that wasn't calculated
    /// for your particular model or dataset.
    /// </para>
    /// </remarks>
    public bool HasMetric(MetricType metricType)
    {
        return metricType switch
        {
            MetricType.ConditionNumber => true,
            MetricType.LogPointwisePredictiveDensity => true,
            MetricType.ObservedTestStatistic => true,
            MetricType.MarginalLikelihood => true,
            MetricType.ReferenceModelMarginalLikelihood => true,
            MetricType.LogLikelihood => true,
            MetricType.EffectiveNumberOfParameters => true,
            MetricType.EuclideanDistance => true,
            MetricType.ManhattanDistance => true,
            MetricType.CosineSimilarity => true,
            MetricType.JaccardSimilarity => true,
            MetricType.HammingDistance => true,
            MetricType.MahalanobisDistance => true,
            MetricType.MutualInformation => true,
            MetricType.NormalizedMutualInformation => true,
            MetricType.VariationOfInformation => true,
            MetricType.SilhouetteScore => true,
            MetricType.CalinskiHarabaszIndex => true,
            MetricType.DaviesBouldinIndex => true,
            MetricType.MeanAveragePrecision => true,
            MetricType.NormalizedDiscountedCumulativeGain => true,
            MetricType.MeanReciprocalRank => true,
            _ => false,
        };
    }
}
