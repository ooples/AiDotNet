using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Association;

/// <summary>
/// Apriori-based Association Rule Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses association rule mining concepts to find features that frequently
/// co-occur with the target class, selecting features with high support
/// and confidence.
/// </para>
/// <para><b>For Beginners:</b> Association rules find patterns like "people who
/// buy bread often buy butter." We apply this to features: features that
/// "associate" strongly with the target class (high confidence) and appear
/// often (high support) are selected.
/// </para>
/// </remarks>
public class AprioriBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minSupport;
    private readonly double _minConfidence;
    private readonly int _nBins;

    private double[]? _associationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? AssociationScores => _associationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public AprioriBasedSelector(
        int nFeaturesToSelect = 10,
        double minSupport = 0.1,
        double minConfidence = 0.5,
        int nBins = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minSupport = minSupport;
        _minConfidence = minConfidence;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "AprioriBasedSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            double yVal = NumOps.ToDouble(target[i]);
            y[i] = yVal > 0.5 ? 1 : 0; // Binarize
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Discretize features
        var discretized = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                min = Math.Min(min, X[i, j]);
                max = Math.Max(max, X[i, j]);
            }
            double range = max - min + 1e-10;
            for (int i = 0; i < n; i++)
                discretized[i, j] = Math.Min((int)((X[i, j] - min) / range * _nBins), _nBins - 1);
        }

        int nPositive = y.Count(yi => yi == 1);
        double targetSupport = (double)nPositive / n;

        _associationScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            double maxScore = 0;

            // Check each bin value
            for (int b = 0; b < _nBins; b++)
            {
                // Count transactions with this feature value
                int featureCount = 0;
                int jointCount = 0;

                for (int i = 0; i < n; i++)
                {
                    if (discretized[i, j] == b)
                    {
                        featureCount++;
                        if (y[i] == 1)
                            jointCount++;
                    }
                }

                if (featureCount == 0) continue;

                double support = (double)jointCount / n;
                double confidence = (double)jointCount / featureCount;
                double lift = confidence / (targetSupport + 1e-10);

                // Score combines support, confidence, and lift
                if (support >= _minSupport && confidence >= _minConfidence)
                {
                    double score = support * confidence * lift;
                    maxScore = Math.Max(maxScore, score);
                }
            }

            _associationScores[j] = maxScore;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _associationScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AprioriBasedSelector has not been fitted.");

        int numRows = data.Rows;
        int numCols = _selectedIndices.Length;
        var result = new T[numRows, numCols];

        for (int i = 0; i < numRows; i++)
            for (int j = 0; j < numCols; j++)
                result[i, j] = data[i, _selectedIndices[j]];

        return new Matrix<T>(result);
    }

    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("AprioriBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("AprioriBasedSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
