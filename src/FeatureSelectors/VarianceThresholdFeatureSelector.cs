namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that removes features with variance below a specified threshold.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix<double>, Tensor<double>, etc.).</typeparam>
/// <remarks>
/// <para>
/// <b>For Beginners:</b> Variance is a measure of how spread out the values of a feature are. Features 
/// with low variance have values that are very similar across all samples, which means they provide 
/// little information for making predictions.
/// </para>
/// <para>
/// This feature selector removes features that don't vary much, keeping only those with variance 
/// above the specified threshold. Think of it like removing a weather forecast that always predicts 
/// "sunny" - it's not helpful because it doesn't change.
/// </para>
/// </remarks>
public class VarianceThresholdFeatureSelector<T, TInput> : IFeatureSelector<T, TInput>
{
    /// <summary>
    /// The variance threshold below which features will be removed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This value determines how much variation a feature must have to be kept. 
    /// Features with variance below this threshold will be removed. The default value is 0.1.
    /// </remarks>
    private readonly T _threshold = default!;
    
    /// <summary>
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private readonly INumericOperations<T> _numOps = default!;

    /// <summary>
    /// The strategy to use for extracting features from higher-dimensional tensors.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When working with complex data like images (which have height, width, and color channels),
    /// we need a strategy to convert these multi-dimensional values into single values that can be analyzed.
    /// This setting controls how that conversion happens.
    /// </remarks>
    private readonly FeatureExtractionStrategy _higherDimensionStrategy = FeatureExtractionStrategy.Mean;

    /// <summary>
    /// Weights to apply when using the WeightedSum feature extraction strategy.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> When combining multiple values from complex data, this dictionary lets you
    /// assign different levels of importance to different parts of your data. Higher weights mean
    /// those parts have more influence on the final value.
    /// </remarks>
    private readonly Dictionary<int, T> _dimensionWeights = [];

    /// <summary>
    /// Initializes a new instance of the VarianceThresholdFeatureSelector class.
    /// </summary>
    /// <param name="threshold">Optional variance threshold value. If not provided, a default value of 0.1 is used.</param>
    /// <param name="higherDimensionStrategy">Strategy to use for extracting features from higher-dimensional tensors.</param>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This constructor creates a new feature selector with either a custom threshold 
    /// or the default threshold of 0.1.
    /// </para>
    /// <para>
    /// The threshold value should be appropriate for your data. A higher threshold will remove more features, 
    /// while a lower threshold will keep more features. The default value of 0.1 is a good starting point, 
    /// but you may need to adjust it based on your specific dataset.
    /// </para>
    /// <para>
    /// The higherDimensionStrategy parameter controls how complex data like images is processed. For most
    /// cases, the default Mean strategy works well.
    /// </para>
    /// </remarks>
    public VarianceThresholdFeatureSelector(
        double threshold = 0.1,
        FeatureExtractionStrategy higherDimensionStrategy = FeatureExtractionStrategy.Mean)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = _numOps.FromDouble(threshold);
        _higherDimensionStrategy = higherDimensionStrategy;
    }

    /// <summary>
    /// Selects features from the input data based on their variance.
    /// </summary>
    /// <param name="allFeatures">The input data containing all potential features.</param>
    /// <returns>Data containing only the features with variance above the threshold.</returns>
    /// <remarks>
    /// <para>
    /// <b>For Beginners:</b> This method analyzes each feature (column) in your dataset and keeps only 
    /// those that have enough variation (variance above the threshold). It works by:
    /// </para>
    /// <para>
    /// 1. Calculating the mean (average) of each feature
    /// </para>
    /// <para>
    /// 2. Calculating the variance of each feature (how spread out the values are)
    /// </para>
    /// <para>
    /// 3. Keeping only the features with variance greater than or equal to the threshold
    /// </para>
    /// <para>
    /// This approach helps remove features that are nearly constant across all samples, which typically 
    /// don't provide useful information for prediction tasks.
    /// </para>
    /// </remarks>
    public TInput SelectFeatures(TInput allFeatures)
    {
        // Get dimensions using the helper methods
        int numSamples = InputHelper<T, TInput>.GetBatchSize(allFeatures);
        int numFeatures = InputHelper<T, TInput>.GetInputSize(allFeatures);
        
        // Track which features to keep
        var selectedFeatureIndices = new List<int>();
        
        // Analyze each feature for variance
        for (int i = 0; i < numFeatures; i++)
        {
            // Extract the feature vector
            var featureVector = FeatureSelectorHelper<T, TInput>.ExtractFeatureVector(
                allFeatures, 
                i, 
                numSamples, 
                _higherDimensionStrategy,
                _dimensionWeights);
            
            // Calculate mean and variance
            var mean = StatisticsHelper<T>.CalculateMean(featureVector);
            var variance = StatisticsHelper<T>.CalculateVariance(featureVector, mean);
            
            // Keep features with variance above threshold
            if (_numOps.GreaterThanOrEquals(variance, _threshold))
            {
                selectedFeatureIndices.Add(i);
            }
        }
        
        // Create result with only the selected features
        return FeatureSelectorHelper<T, TInput>.CreateFilteredData(allFeatures, selectedFeatureIndices);
    }
}