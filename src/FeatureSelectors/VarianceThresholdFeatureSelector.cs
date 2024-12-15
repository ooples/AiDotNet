namespace AiDotNet.FeatureSelectors;

public class VarianceThresholdFeatureSelector<T> : IFeatureSelector<T>
{
    private readonly T _threshold;
    private readonly INumericOperations<T> _numOps;

    public VarianceThresholdFeatureSelector(T? threshold = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold ?? GetDefaultThreshold();
    }

    private T GetDefaultThreshold()
    {
        return _numOps.FromDouble(0.1); // 10% of the maximum variance
    }

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

        return new Matrix<T>(selectedFeatures, _numOps);
    }
}