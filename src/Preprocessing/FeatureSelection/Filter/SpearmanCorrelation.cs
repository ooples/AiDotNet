using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Spearman rank correlation for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Measures the monotonic relationship between features and target using ranks
/// instead of raw values. Robust to outliers and non-linear monotonic relationships.
/// </para>
/// <para><b>For Beginners:</b> While Pearson correlation only detects linear relationships,
/// Spearman detects any monotonic relationship (consistently increasing or decreasing).
/// It works by ranking values, so outliers have less impact. Perfect for ordinal data
/// or when relationships are curved but consistent.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpearmanCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _minCorrelation;

    private double[]? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Correlations => _correlations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SpearmanCorrelation(
        int nFeaturesToSelect = 10,
        double minCorrelation = 0.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _minCorrelation = minCorrelation;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SpearmanCorrelation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Rank the target
        var targetRanks = ComputeRanks(target, n);

        _correlations = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Rank the feature
            var featureRanks = ComputeFeatureRanks(data, j, n);

            // Compute Spearman correlation (Pearson on ranks)
            _correlations[j] = ComputePearsonOnRanks(featureRanks, targetRanks, n);
        }

        // Select features above threshold or top by absolute correlation
        var candidates = new List<int>();
        for (int j = 0; j < p; j++)
            if (Math.Abs(_correlations[j]) >= _minCorrelation)
                candidates.Add(j);

        if (candidates.Count >= _nFeaturesToSelect)
        {
            _selectedIndices = candidates
                .OrderByDescending(j => Math.Abs(_correlations[j]))
                .Take(_nFeaturesToSelect)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = _correlations
                .Select((c, idx) => (Corr: Math.Abs(c), Index: idx))
                .OrderByDescending(x => x.Corr)
                .Take(_nFeaturesToSelect)
                .Select(x => x.Index)
                .OrderBy(x => x)
                .ToArray();
        }

        IsFitted = true;
    }

    private double[] ComputeRanks(Vector<T> values, int n)
    {
        var valuesWithIndices = new List<(double Value, int Index)>();
        for (int i = 0; i < n; i++)
            valuesWithIndices.Add((NumOps.ToDouble(values[i]), i));

        var sorted = valuesWithIndices.OrderBy(x => x.Value).ToList();
        var ranks = new double[n];

        int i2 = 0;
        while (i2 < n)
        {
            int j2 = i2;
            while (j2 < n && Math.Abs(sorted[j2].Value - sorted[i2].Value) < 1e-10)
                j2++;

            double avgRank = (i2 + j2 + 1) / 2.0;
            for (int m = i2; m < j2; m++)
                ranks[sorted[m].Index] = avgRank;

            i2 = j2;
        }

        return ranks;
    }

    private double[] ComputeFeatureRanks(Matrix<T> data, int featureIdx, int n)
    {
        var valuesWithIndices = new List<(double Value, int Index)>();
        for (int i = 0; i < n; i++)
            valuesWithIndices.Add((NumOps.ToDouble(data[i, featureIdx]), i));

        var sorted = valuesWithIndices.OrderBy(x => x.Value).ToList();
        var ranks = new double[n];

        int i2 = 0;
        while (i2 < n)
        {
            int j2 = i2;
            while (j2 < n && Math.Abs(sorted[j2].Value - sorted[i2].Value) < 1e-10)
                j2++;

            double avgRank = (i2 + j2 + 1) / 2.0;
            for (int m = i2; m < j2; m++)
                ranks[sorted[m].Index] = avgRank;

            i2 = j2;
        }

        return ranks;
    }

    private double ComputePearsonOnRanks(double[] x, double[] y, int n)
    {
        double xMean = x.Average();
        double yMean = y.Average();

        double covariance = 0, xVar = 0, yVar = 0;
        for (int i = 0; i < n; i++)
        {
            double xDiff = x[i] - xMean;
            double yDiff = y[i] - yMean;
            covariance += xDiff * yDiff;
            xVar += xDiff * xDiff;
            yVar += yDiff * yDiff;
        }

        double denom = Math.Sqrt(xVar * yVar);
        return denom > 1e-10 ? covariance / denom : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpearmanCorrelation has not been fitted.");

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
        throw new NotSupportedException("SpearmanCorrelation does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpearmanCorrelation has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
