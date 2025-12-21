namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that removes features with variance below a specified threshold.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
/// <typeparam name="TInput">The input data type (Matrix, Tensor, etc.).</typeparam>
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
public class VarianceThresholdFeatureSelector<T, TInput> : FeatureSelectorBase<T, TInput>
{
    /// <summary>
    /// The variance threshold below which features will be removed.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This value determines how much variation a feature must have to be kept.
    /// Features with variance below this threshold will be removed. The default value is 0.1.
    /// </remarks>
    private readonly T _threshold;

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
        : base(higherDimensionStrategy)
    {
        _threshold = NumOps.FromDouble(threshold);
    }

    /// <summary>
    /// Determines which features to select based on their variance.
    /// </summary>
    /// <param name="allFeatures">The input data containing all potential features.</param>
    /// <param name="numSamples">The number of samples in the dataset.</param>
    /// <param name="numFeatures">The total number of features in the dataset.</param>
    /// <returns>A list of indices representing the selected features.</returns>
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
    protected override List<int> SelectFeatureIndices(TInput allFeatures, int numSamples, int numFeatures)
    {
        // Track which features to keep
        var selectedFeatureIndices = new List<int>();

        // Analyze each feature for variance
        for (int i = 0; i < numFeatures; i++)
        {
            // Extract the feature vector using the base class helper
            var featureVector = ExtractFeatureVector(allFeatures, i, numSamples);

            // Calculate mean and variance
            var mean = StatisticsHelper<T>.CalculateMean(featureVector);
            var variance = StatisticsHelper<T>.CalculateVariance(featureVector, mean);

            // Keep features with variance above threshold
            if (NumOps.GreaterThanOrEquals(variance, _threshold))
            {
                selectedFeatureIndices.Add(i);
            }
        }

        return selectedFeatureIndices;
    }
}
