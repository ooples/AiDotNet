namespace AiDotNet.FeatureSelectors;

/// <summary>
/// A feature selector that removes features with variance below a specified threshold.
/// </summary>
/// <typeparam name="T">The data type used for calculations (typically float or double).</typeparam>
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
public class VarianceThresholdFeatureSelector<T> : IFeatureSelector<T>
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
    /// Provides operations for numeric calculations with type T.
    /// </summary>
    /// <remarks>
    /// <b>For Beginners:</b> This is a helper object that knows how to perform math operations 
    /// on the specific number type you're using (like float or double).
    /// </remarks>
    private readonly INumericOperations<T> _numOps;

    /// <summary>
    /// Initializes a new instance of the VarianceThresholdFeatureSelector class.
    /// </summary>
    /// <param name="threshold">Optional variance threshold value. If not provided, a default value of 0.1 is used.</param>
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
    /// </remarks>
    public VarianceThresholdFeatureSelector(T? threshold = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold ?? GetDefaultThreshold();
    }

    /// <summary>
    /// Gets the default variance threshold value.
    /// </summary>
    /// <returns>The default threshold value of 0.1.</returns>
    /// <remarks>
    /// <b>For Beginners:</b> This private method simply returns the default threshold value of 0.1 
    /// when no custom threshold is provided.
    /// </remarks>
    private T GetDefaultThreshold()
    {
        return _numOps.FromDouble(0.1); // 10% of the maximum variance
    }

    /// <summary>
    /// Selects features from the input matrix based on their variance.
    /// </summary>
    /// <param name="allFeaturesMatrix">The matrix containing all potential features.</param>
    /// <returns>A matrix containing only the features with variance above the threshold.</returns>
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
    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        var selectedFeatures = new List<Vector<T>>();
        var numFeatures = allFeaturesMatrix.Columns;

        for (int i = 0; i < numFeatures; i++)
        {
            var feature = allFeaturesMatrix.GetColumn(i);
            var mean = StatisticsHelper<T>.CalculateMean(feature);
            var variance = StatisticsHelper<T>.CalculateVariance(feature, mean);

            if (_numOps.GreaterThanOrEquals(variance, _threshold))
            {
                selectedFeatures.Add(feature);
            }
        }

        return new Matrix<T>(selectedFeatures);
    }
}