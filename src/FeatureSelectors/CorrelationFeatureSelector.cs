namespace AiDotNet.FeatureSelectors;

public class CorrelationFeatureSelector<T> : IFeatureSelector<T>
{
    private readonly T _threshold;
    private readonly INumericOperations<T> _numOps;

    public CorrelationFeatureSelector(T? threshold = default)
    {
        _numOps = MathHelper.GetNumericOperations<T>();
        _threshold = threshold ?? GetDefaultThreshold();
    }

    private T GetDefaultThreshold()
    {
        return _numOps.FromDouble(0.5);
    }

    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        var selectedFeatures = new List<Vector<T>>();
        var numFeatures = allFeaturesMatrix.Columns;

        for (int i = 0; i < numFeatures; i++)
        {
            bool isIndependent = true;
            var featureI = allFeaturesMatrix.GetColumn(i);

            for (int j = 0; j < selectedFeatures.Count; j++)
            {
                var correlation = StatisticsHelper<T>.CalculatePearsonCorrelation(featureI, selectedFeatures[j]);
                if (_numOps.GreaterThan(_numOps.Abs(correlation), _threshold))
                {
                    isIndependent = false;
                    break;
                }
            }

            if (isIndependent)
            {
                selectedFeatures.Add(featureI);
            }
        }

        return new Matrix<T>(selectedFeatures, _numOps);
    }
}