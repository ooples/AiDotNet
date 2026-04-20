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
    private Matrix<T> _correlationMatrix = default!;
    public Matrix<T> CorrelationMatrix { get { EnsureFullStatsComputed(); return _correlationMatrix; } private set { _correlationMatrix = value; } }

    /// <summary>
    /// Gets the covariance matrix showing how features vary together.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This matrix shows how features change together.
    /// It's similar to the correlation matrix but uses a different scale.
    /// It helps identify patterns in how your features behave together.
    /// </para>
    /// </remarks>
    private Matrix<T> _covarianceMatrix = default!;
    public Matrix<T> CovarianceMatrix { get { EnsureFullStatsComputed(); return _covarianceMatrix; } private set { _covarianceMatrix = value; } }

    /// <summary>
    /// Gets the Variance Inflation Factor (VIF) for each feature.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> VIF helps identify if some features are too similar to others.
    /// High VIF values (usually above 5 or 10) suggest that a feature might be redundant,
    /// as its information is already captured by other features.
    /// </para>
    /// </remarks>
    private List<T> _vIFList = default!;
    public List<T> VIFList { get { EnsureFullStatsComputed(); return _vIFList; } private set { _vIFList = value; } }

    /// <summary>
    /// Gets the condition number, a measure of the model's numerical stability.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> The condition number tells you if small changes in your data
    /// might cause big changes in your model's predictions. A high condition number
    /// (typically above 30) suggests that your model might be unstable and sensitive to small data changes.
    /// </para>
    /// </remarks>
    private T _conditionNumber = default!;
    public T ConditionNumber { get { EnsureFullStatsComputed(); return _conditionNumber; } private set { _conditionNumber = value; } }

    /// <summary>
    /// Gets the log pointwise predictive density, a measure of prediction accuracy.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a way to measure how well your model's predictions match the actual data.
    /// Higher values generally indicate better predictions. It's particularly useful when comparing different models.
    /// </para>
    /// </remarks>
    private T _logPointwisePredictiveDensity = default!;
    public T LogPointwisePredictiveDensity { get { EnsureFullStatsComputed(); return _logPointwisePredictiveDensity; } private set { _logPointwisePredictiveDensity = value; } }

    /// <summary>
    /// Gets the leave-one-out predictive densities for each data point.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This shows how well the model predicts each data point when it's trained without that point.
    /// It helps identify which data points might be harder for the model to predict accurately.
    /// </para>
    /// </remarks>
    private List<T> _leaveOneOutPredictiveDensities = default!;
    public List<T> LeaveOneOutPredictiveDensities { get { EnsureFullStatsComputed(); return _leaveOneOutPredictiveDensities; } private set { _leaveOneOutPredictiveDensities = value; } }

    /// <summary>
    /// Gets the observed test statistic for model evaluation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a single number that summarizes how well your model fits the data.
    /// It's used in statistical tests to determine if your model is significantly better than a simpler alternative.
    /// </para>
    /// </remarks>
    private T _observedTestStatistic = default!;
    public T ObservedTestStatistic { get { EnsureFullStatsComputed(); return _observedTestStatistic; } private set { _observedTestStatistic = value; } }

    /// <summary>
    /// Gets samples from the posterior predictive distribution.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> These are possible predictions your model might make if you ran it multiple times.
    /// They help you understand the range and uncertainty of your model's predictions.
    /// </para>
    /// </remarks>
    private List<T> _posteriorPredictiveSamples = default!;
    public List<T> PosteriorPredictiveSamples { get { EnsureFullStatsComputed(); return _posteriorPredictiveSamples; } private set { _posteriorPredictiveSamples = value; } }

    /// <summary>
    /// Gets the marginal likelihood of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is a measure of how well your model fits the data, taking into account its complexity.
    /// It helps in comparing different models, with higher values generally indicating better models.
    /// </para>
    /// </remarks>
    private T _marginalLikelihood = default!;
    public T MarginalLikelihood { get { EnsureFullStatsComputed(); return _marginalLikelihood; } private set { _marginalLikelihood = value; } }

    /// <summary>
    /// Gets the marginal likelihood of a reference (simpler) model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is the marginal likelihood for a basic, simple model.
    /// It's used as a comparison point to see how much better your more complex model performs.
    /// </para>
    /// </remarks>
    private T _referenceModelMarginalLikelihood = default!;
    public T ReferenceModelMarginalLikelihood { get { EnsureFullStatsComputed(); return _referenceModelMarginalLikelihood; } private set { _referenceModelMarginalLikelihood = value; } }

    /// <summary>
    /// Gets the log-likelihood of the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how probable your data is under your model.
    /// Higher values mean your model fits the data better. It's often used in more advanced statistical techniques.
    /// </para>
    /// </remarks>
    private T _logLikelihood = default!;
    public T LogLikelihood { get { EnsureFullStatsComputed(); return _logLikelihood; } private set { _logLikelihood = value; } }

    /// <summary>
    /// Gets the effective number of parameters in the model.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This estimates how complex your model is in practice.
    /// It might be different from the actual number of parameters and helps identify if your model is overfitting
    /// (using more complexity than needed to explain the data).
    /// </para>
    /// </remarks>
    private T _effectiveNumberOfParameters = default!;
    public T EffectiveNumberOfParameters { get { EnsureFullStatsComputed(); return _effectiveNumberOfParameters; } private set { _effectiveNumberOfParameters = value; } }

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
    private T _euclideanDistance = default!;
    public T EuclideanDistance { get { EnsureFullStatsComputed(); return _euclideanDistance; } private set { _euclideanDistance = value; } }

    /// <summary>
    /// Gets the Manhattan distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the distance between actual and predicted values as if you could only move
    /// horizontally or vertically (like navigating city blocks). Lower values indicate better predictions.
    /// </para>
    /// </remarks>
    private T _manhattanDistance = default!;
    public T ManhattanDistance { get { EnsureFullStatsComputed(); return _manhattanDistance; } private set { _manhattanDistance = value; } }

    /// <summary>
    /// Gets the cosine similarity between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how similar the direction of your predictions is to the actual values,
    /// ignoring their magnitude. Values closer to 1 indicate more similar directions.
    /// </para>
    /// </remarks>
    private T _cosineSimilarity = default!;
    public T CosineSimilarity { get { EnsureFullStatsComputed(); return _cosineSimilarity; } private set { _cosineSimilarity = value; } }

    /// <summary>
    /// Gets the Jaccard similarity between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures the overlap between your actual and predicted values.
    /// It's especially useful for binary (yes/no) predictions. Values closer to 1 indicate more overlap.
    /// </para>
    /// </remarks>
    private T _jaccardSimilarity = default!;
    public T JaccardSimilarity { get { EnsureFullStatsComputed(); return _jaccardSimilarity; } private set { _jaccardSimilarity = value; } }

    /// <summary>
    /// Gets the Hamming distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This counts how many predictions are different from the actual values.
    /// It's most useful for categorical predictions. Lower values indicate fewer differences.
    /// </para>
    /// </remarks>
    private T _hammingDistance = default!;
    public T HammingDistance { get { EnsureFullStatsComputed(); return _hammingDistance; } private set { _hammingDistance = value; } }

    /// <summary>
    /// Gets the Mahalanobis distance between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is an advanced distance measure that takes into account how your features
    /// are related to each other. It can be more meaningful than simpler distances when your features are correlated.
    /// </para>
    /// </remarks>
    private T _mahalanobisDistance = default!;
    public T MahalanobisDistance { get { EnsureFullStatsComputed(); return _mahalanobisDistance; } private set { _mahalanobisDistance = value; } }

    /// <summary>
    /// Gets the mutual information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how much information your predictions provide about the actual values.
    /// Higher values mean your predictions are more informative and closely related to the actual values.
    /// </para>
    /// </remarks>
    private T _mutualInformation = default!;
    public T MutualInformation { get { EnsureFullStatsComputed(); return _mutualInformation; } private set { _mutualInformation = value; } }

    /// <summary>
    /// Gets the normalized mutual information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This is similar to mutual information, but scaled to be between 0 and 1.
    /// It's easier to interpret across different datasets. Values closer to 1 indicate better predictions.
    /// </para>
    /// </remarks>
    private T _normalizedMutualInformation = default!;
    public T NormalizedMutualInformation { get { EnsureFullStatsComputed(); return _normalizedMutualInformation; } private set { _normalizedMutualInformation = value; } }

    /// <summary>
    /// Gets the variation of information between actual and predicted values.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how different your predictions are from the actual values.
    /// Lower values indicate that your predictions are more similar to the actual values.
    /// It's particularly useful when comparing different clustering results.
    /// </para>
    /// </remarks>
    private T _variationOfInformation = default!;
    public T VariationOfInformation { get { EnsureFullStatsComputed(); return _variationOfInformation; } private set { _variationOfInformation = value; } }

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
    private T _silhouetteScore = default!;
    public T SilhouetteScore { get { EnsureFullStatsComputed(); return _silhouetteScore; } private set { _silhouetteScore = value; } }

    /// <summary>
    /// Gets the Calinski-Harabasz index, a measure of cluster separation.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index tells you how well-separated your groups (clusters) are.
    /// Higher values mean your groups are more distinct from each other, which is generally better.
    /// It's useful when comparing different ways of grouping your data.
    /// </para>
    /// </remarks>
    private T _calinskiHarabaszIndex = default!;
    public T CalinskiHarabaszIndex { get { EnsureFullStatsComputed(); return _calinskiHarabaszIndex; } private set { _calinskiHarabaszIndex = value; } }

    /// <summary>
    /// Gets the Davies-Bouldin index, a measure of the average similarity between each cluster and its most similar cluster.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This index helps you understand how well-separated your groups (clusters) are.
    /// Lower values are better, meaning your groups are more distinct from each other.
    /// It's particularly useful when you're not sure how many groups to divide your data into.
    /// </para>
    /// </remarks>
    private T _daviesBouldinIndex = default!;
    public T DaviesBouldinIndex { get { EnsureFullStatsComputed(); return _daviesBouldinIndex; } private set { _daviesBouldinIndex = value; } }

    /// <summary>
    /// Gets the Mean Average Precision, a measure of ranking quality.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model ranks items, especially in search or recommendation systems.
    /// It ranges from 0 to 1, where 1 is perfect. It considers both the order of your predictions and their accuracy.
    /// For example, in a search engine, it would measure how well the most relevant results are placed at the top.
    /// </para>
    /// </remarks>
    private T _meanAveragePrecision = default!;
    public T MeanAveragePrecision { get { EnsureFullStatsComputed(); return _meanAveragePrecision; } private set { _meanAveragePrecision = value; } }

    /// <summary>
    /// Gets the Normalized Discounted Cumulative Gain, a measure of ranking quality that takes the position of correct items into account.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model ranks items, giving more importance to correct predictions at the top of the list.
    /// It ranges from 0 to 1, where 1 is perfect. It's often used in search engines or recommendation systems to ensure the most relevant items appear first.
    /// </para>
    /// </remarks>
    private T _normalizedDiscountedCumulativeGain = default!;
    public T NormalizedDiscountedCumulativeGain { get { EnsureFullStatsComputed(); return _normalizedDiscountedCumulativeGain; } private set { _normalizedDiscountedCumulativeGain = value; } }

    /// <summary>
    /// Gets the Mean Reciprocal Rank, a statistic measuring the performance of a system that produces a list of possible responses to a query.
    /// </summary>
    /// <remarks>
    /// <para><b>For Beginners:</b> This measures how well your model places the first correct answer in a list of predictions.
    /// It ranges from 0 to 1, where 1 means the correct answer is always first. It's often used in question-answering systems
    /// or search engines to measure how quickly a user might find the right answer.
    /// </para>
    /// </remarks>
    private T _meanReciprocalRank = default!;
    public T MeanReciprocalRank { get { EnsureFullStatsComputed(); return _meanReciprocalRank; } private set { _meanReciprocalRank = value; } }

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
        VIFList = new List<T>();
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
        LeaveOneOutPredictiveDensities = new List<T>();
        ObservedTestStatistic = _numOps.Zero;
        PosteriorPredictiveSamples = new List<T>();
        MarginalLikelihood = _numOps.Zero;
        ReferenceModelMarginalLikelihood = _numOps.Zero;
        LogLikelihood = _numOps.Zero;
        EffectiveNumberOfParameters = _numOps.Zero;
        Actual = inputs.Actual;
        Predicted = inputs.Predicted;
        Features = inputs.XMatrix;
        Model = inputs.Model;
        FeatureNames = inputs.FeatureNames ?? new List<string>();
        FeatureValues = inputs.FeatureValues ?? new Dictionary<string, TOutput>();

        // Defer expensive computation (CorrelationMatrix, VIF, ConditionNumber,
        // LOO, posterior sampling, etc.) until first property access.
        if (!IsEmptyInput(inputs))
        {
            _deferredInputs = inputs;
        }
    }

    private ModelStatsInputs<T, TInput, TOutput>? _deferredInputs;
    private bool _fullStatsComputed;

    private void EnsureFullStatsComputed()
    {
        if (_fullStatsComputed || _deferredInputs is null) return;

        // Run the heavy computation BEFORE marking computed, so a throw inside
        // CalculateModelStats lets the next property access retry instead of leaving
        // the instance permanently stuck with default values.
        CalculateModelStats(_deferredInputs);

        _fullStatsComputed = true;
        _deferredInputs = null;
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
        // All intermediate values held in locals until the end. Previous code
        // read CorrelationMatrix and CovarianceMatrix through their property
        // getters during computation; those getters call
        // EnsureFullStatsComputed, which re-enters this method — unbounded
        // recursion + StackOverflowException. Same bug class as BasicStats /
        // ErrorStats / PredictionStats.
        Matrix<T> matrix = ConversionsHelper.ConvertToMatrix<T, TInput>(inputs.XMatrix);
        Vector<T> actual = ConversionsHelper.ConvertToVector<T, TOutput>(inputs.Actual);
        Vector<T> predicted = ConversionsHelper.ConvertToVector<T, TOutput>(inputs.Predicted);

        Vector<T> coefficients = Vector<T>.Empty();
        if (inputs.Coefficients != null)
        {
            coefficients = ConversionsHelper.ConvertToVector<T, object>(inputs.Coefficients);
        }

        var featureCount = inputs.FeatureCount;

        var correlationMatrix = StatisticsHelper<T>.CalculateCorrelationMatrix(matrix, _options);
        var covarianceMatrix = StatisticsHelper<T>.CalculateCovarianceMatrix(matrix);
        var vifList = StatisticsHelper<T>.CalculateVIF(correlationMatrix, _options);
        T conditionNumber = StatisticsHelper<T>.CalculateConditionNumber(matrix, _options);
        T logPpd = StatisticsHelper<T>.CalculateLogPointwisePredictiveDensity(actual, predicted);

        List<T>? looPd = null;
        if (inputs.FitFunction != null)
        {
            var convertedFitFunction = ConversionsHelper.ConvertFitFunction<T, TInput, TOutput>(inputs.FitFunction);
            Vector<T> wrappedFitFunction(Matrix<T> m, Vector<T> v) => convertedFitFunction(m);
            looPd = StatisticsHelper<T>.CalculateLeaveOneOutPredictiveDensities(matrix, actual, wrappedFitFunction);
        }

        T observedTest = StatisticsHelper<T>.CalculateObservedTestStatistic(actual, predicted);
        var posteriorSamples = StatisticsHelper<T>.CalculatePosteriorPredictiveSamples(actual, predicted, featureCount);
        T marginalLik = StatisticsHelper<T>.CalculateMarginalLikelihood(actual, predicted, featureCount);
        T refModelML = StatisticsHelper<T>.CalculateReferenceModelMarginalLikelihood(actual);
        T logLik = StatisticsHelper<T>.CalculateLogLikelihood(actual, predicted);
        T effParams = StatisticsHelper<T>.CalculateEffectiveNumberOfParameters(matrix, coefficients);
        T mi = StatisticsHelper<T>.CalculateMutualInformation(actual, predicted);
        T nmi = StatisticsHelper<T>.CalculateNormalizedMutualInformation(actual, predicted);
        T voi = StatisticsHelper<T>.CalculateVariationOfInformation(actual, predicted);
        T silhouette = StatisticsHelper<T>.CalculateSilhouetteScore(matrix, predicted);
        T ch = StatisticsHelper<T>.CalculateCalinskiHarabaszIndex(matrix, predicted);
        T db = StatisticsHelper<T>.CalculateDaviesBouldinIndex(matrix, predicted);
        T mAP = StatisticsHelper<T>.CalculateMeanAveragePrecision(actual, predicted, _options.MapTopK);
        T ndcg = StatisticsHelper<T>.CalculateNDCG(actual, predicted, _options.NdcgTopK);
        T mrr = StatisticsHelper<T>.CalculateMeanReciprocalRank(actual, predicted);

        var residuals = StatisticsHelper<T>.CalculateResiduals(actual, predicted);
        var acf = StatisticsHelper<T>.CalculateAutoCorrelationFunction(residuals, _options.AcfMaxLag);
        var pacf = StatisticsHelper<T>.CalculatePartialAutoCorrelationFunction(residuals, _options.PacfMaxLag);

        T euclidean = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Euclidean);
        T manhattan = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Manhattan);
        T cosine = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Cosine);
        T jaccard = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Jaccard);
        T hamming = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Hamming);

        T mahalanobis = _numOps.Zero;
        try
        {
            if (covarianceMatrix.Rows == actual.Length && covarianceMatrix.Columns == actual.Length)
            {
                mahalanobis = StatisticsHelper<T>.CalculateDistance(actual, predicted, DistanceMetricType.Mahalanobis, covarianceMatrix);
            }
        }
        catch (ArgumentException)
        {
            // Silently skip Mahalanobis distance calculation when dimensions don't match.
        }

        // Assign properties once every dependency is a local — no re-entry.
        CorrelationMatrix = correlationMatrix;
        CovarianceMatrix = covarianceMatrix;
        VIFList = vifList;
        ConditionNumber = conditionNumber;
        LogPointwisePredictiveDensity = logPpd;
        if (looPd != null) LeaveOneOutPredictiveDensities = looPd;
        ObservedTestStatistic = observedTest;
        PosteriorPredictiveSamples = posteriorSamples;
        MarginalLikelihood = marginalLik;
        ReferenceModelMarginalLikelihood = refModelML;
        LogLikelihood = logLik;
        EffectiveNumberOfParameters = effParams;
        MutualInformation = mi;
        NormalizedMutualInformation = nmi;
        VariationOfInformation = voi;
        SilhouetteScore = silhouette;
        CalinskiHarabaszIndex = ch;
        DaviesBouldinIndex = db;
        MeanAveragePrecision = mAP;
        NormalizedDiscountedCumulativeGain = ndcg;
        MeanReciprocalRank = mrr;
        AutoCorrelationFunction = acf;
        PartialAutoCorrelationFunction = pacf;
        EuclideanDistance = euclidean;
        ManhattanDistance = manhattan;
        CosineSimilarity = cosine;
        JaccardSimilarity = jaccard;
        HammingDistance = hamming;
        MahalanobisDistance = mahalanobis;
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
