namespace AiDotNet.FeatureSelectors;

/// <summary>
/// Abstract base class for feature selectors that provides common functionality.
/// </summary>
/// <typeparam name="T">The numeric type used for calculations (e.g., double, float).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> This base class provides common functionality that all feature selectors need,
/// such as handling different numeric types and working with complex data structures.
/// </para>
/// <para>
/// By inheriting from this class, specific feature selection implementations can focus on their unique
/// selection logic while reusing common code for data extraction and filtering.
/// </para>
/// </remarks>
public abstract class FeatureSelectorBase<T, TInput> : IFeatureSelector<T, TInput>
{
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This helper object knows how to perform math operations
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    protected readonly INumericOperations<T> NumOps;

    /// <summary>
    /// Gets the global execution engine for vector operations.
    /// </summary>
    protected IEngine Engine => AiDotNetEngine.Current;

    /// <summary>
    /// The strategy to use for extracting features from higher-dimensional tensors.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When working with complex data like images (which have height, width, and color channels),
    /// we need a strategy to convert these multi-dimensional values into single values that can be analyzed.
    /// This setting controls how that conversion happens.
    /// </remarks>
    protected readonly FeatureExtractionStrategy HigherDimensionStrategy;

    /// <summary>
    /// Weights to apply when using the WeightedSum feature extraction strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When combining multiple values from complex data, this dictionary lets you
    /// assign different levels of importance to different parts of your data. Higher weights mean
    /// those parts have more influence on the final value.
    /// </remarks>
    protected readonly Dictionary<int, T> DimensionWeights;

    /// <summary>
    /// Initializes a new instance of the FeatureSelectorBase class.
    /// </summary>
    /// <param name="higherDimensionStrategy">Strategy to use for extracting features from higher-dimensional tensors.</param>
    /// <param name="dimensionWeights">Optional weights for the WeightedSum strategy.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor sets up the common components that all feature selectors need.
    /// </para>
    /// <para>
    /// The higherDimensionStrategy parameter controls how complex data like images is processed. For most
    /// cases, the default Mean strategy works well.
    /// </para>
    /// </remarks>
    protected FeatureSelectorBase(
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean,
        Dictionary<int, T>? dimensionWeights = null)
    {
        NumOps = MathHelper.GetNumericOperations<T>();
        HigherDimensionStrategy = higherDimensionStrategy;
        DimensionWeights = dimensionWeights ?? [];
    }

    /// <summary>
    /// Selects features from the input data based on the specific selection criteria implemented by derived classes.
    /// </summary>
    /// <param name="allFeaturesMatrix">The input data containing all potential features.</param>
    /// <returns>Data containing only the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes your dataset and keeps only the important features (columns).
    /// </para>
    /// <para>
    /// It works by:
    /// 1. Extracting information about the data dimensions (rows and columns)
    /// 2. Calling a specialized method to determine which features are important
    /// 3. Creating a new dataset with only the selected features
    /// </para>
    /// <para>
    /// Different feature selectors implement different strategies for determining which features to keep.
    /// </para>
    /// </remarks>
    public TInput SelectFeatures(TInput allFeaturesMatrix)
    {
        // Get dimensions using the helper methods
        int numSamples = InputHelper<T, TInput>.GetBatchSize(allFeaturesMatrix);
        int numFeatures = InputHelper<T, TInput>.GetInputSize(allFeaturesMatrix);

        // Validate input has features
        if (numFeatures == 0)
        {
            throw new ArgumentException(
                "Input data has no features to select from. Ensure your data has at least one feature (column).",
                nameof(allFeaturesMatrix));
        }

        // Determine which features to select using the derived class's logic
        var selectedFeatureIndices = SelectFeatureIndices(allFeaturesMatrix, numSamples, numFeatures);

        // Safety check: ensure at least one feature is selected
        // Individual selectors should handle this, but this provides a safety net
        if (selectedFeatureIndices.Count == 0)
        {
            throw new InvalidOperationException(
                "No features passed the selection criteria. This typically means the threshold is too strict. " +
                "Consider lowering the threshold value or using a different feature selection strategy. " +
                $"Total features analyzed: {numFeatures}.");
        }

        // Create result with only the selected features
        return FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeaturesMatrix, selectedFeatureIndices);
    }

    /// <summary>
    /// Determines which feature indices should be selected based on the specific selection criteria.
    /// </summary>
    /// <param name="allFeatures">The input data containing all potential features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features in the dataset.</param>
    /// <returns>A list of indices representing the selected features.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This abstract method is where each specific feature selector implements its unique
    /// logic for deciding which features to keep.
    /// </para>
    /// <para>
    /// For example:
    /// - A variance-based selector might calculate variance for each feature and keep those above a threshold
    /// - A correlation-based selector might remove features that are too similar to each other
    /// - A statistical selector might use statistical tests to identify important features
    /// </para>
    /// </remarks>
    protected abstract List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures);

    /// <summary>
    /// Extracts a single feature vector from the input data.
    /// </summary>
    /// <param name="allFeatures">The input data containing all features.</param>
    /// <param name="featureIndex">The index of the feature to extract.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <returns>A vector containing the values of the specified feature across all samples.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This helper method extracts a single column (feature) from your data,
    /// giving you all the values for that feature across all samples. This is useful for analyzing
    /// individual features.
    /// </remarks>
    protected Vector<T> ExtractFeatureVector(TInput allFeatures, int featureIndex, int numSamples)
    {
        return FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(
            allFeatures,
            featureIndex,
            numSamples,
            HigherDimensionStrategy,
            DimensionWeights);
    }
}
