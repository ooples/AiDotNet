namespace AiDotNet.FeatureSelectors;

public class RecursiveFeatureElimination<T> : IFeatureSelector<T>
{
    private readonly int _numFeaturesToSelect;
    private readonly IRegression<T> _model;
    private readonly INumericOperations<T> _numOps;

    public RecursiveFeatureElimination(IRegression<T> model, int? numFeaturesToSelect = null)
    {
        _model = model;
        _numOps = MathHelper.GetNumericOperations<T>();
        _numFeaturesToSelect = numFeaturesToSelect ?? GetDefaultNumFeatures();
    }

    private int GetDefaultNumFeatures()
    {
        return Math.Max(1, (int)(_model.Coefficients.Length * 0.5)); // Default to 50% of features
    }

    public Matrix<T> SelectFeatures(Matrix<T> allFeaturesMatrix)
    {
        var numFeatures = allFeaturesMatrix.Columns;
        var featureIndices = Enumerable.Range(0, numFeatures).ToList();
        var selectedFeatures = new List<int>();

        while (selectedFeatures.Count < _numFeaturesToSelect && featureIndices.Count > 0)
        {
            var subMatrix = new Matrix<T>(allFeaturesMatrix.Rows, featureIndices.Count);
            for (int i = 0; i < featureIndices.Count; i++)
            {
                subMatrix.SetColumn(i, allFeaturesMatrix.GetColumn(featureIndices[i]));
            }

            // Assuming we have a dummy target vector for feature importance calculation
            var dummyTarget = new Vector<T>(allFeaturesMatrix.Rows);
            _model.Train(subMatrix, dummyTarget);

            var featureImportances = _model.Coefficients.Select((c, i) => (_numOps.Abs(c), i)).ToList();
            featureImportances.Sort((a, b) => _numOps.GreaterThan(b.Item1, a.Item1) ? -1 : (_numOps.Equals(b.Item1, a.Item1) ? 0 : 1));

            var leastImportantFeatureIndex = featureImportances.Last().i;
            selectedFeatures.Insert(0, featureIndices[leastImportantFeatureIndex]);
            featureIndices.RemoveAt(leastImportantFeatureIndex);
        }

        var result = new Matrix<T>(allFeaturesMatrix.Rows, _numFeaturesToSelect);
        for (int i = 0; i < _numFeaturesToSelect; i++)
        {
            result.SetColumn(i, allFeaturesMatrix.GetColumn(selectedFeatures[i]));
        }

        return result;
    }
}