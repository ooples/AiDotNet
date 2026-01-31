using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Spearman Rank Correlation-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Spearman Correlation measures the monotonic relationship between features
/// and target using ranks instead of raw values. It can capture non-linear
/// monotonic relationships that Pearson correlation would miss.
/// </para>
/// <para><b>For Beginners:</b> Instead of measuring how well a straight line fits,
/// Spearman correlation checks if the feature and target increase together (or one
/// increases while the other decreases). It works with rankings, so it's robust to
/// outliers and can detect curved relationships that are still consistently going
/// up or down.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class SpearmanCorrelation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CorrelationScores => _correlationScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SpearmanCorrelation(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
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

        _correlationScores = new double[p];

        // Compute target ranks
        var targetRanks = ComputeRanks(target, n);

        for (int j = 0; j < p; j++)
        {
            // Compute feature ranks
            var featureRanks = ComputeFeatureRanks(data, j, n);

            // Compute Pearson correlation on ranks
            double xMean = 0, yMean = 0;
            for (int i = 0; i < n; i++)
            {
                xMean += featureRanks[i];
                yMean += targetRanks[i];
            }
            xMean /= n;
            yMean /= n;

            double sxy = 0, sxx = 0, syy = 0;
            for (int i = 0; i < n; i++)
            {
                double xDiff = featureRanks[i] - xMean;
                double yDiff = targetRanks[i] - yMean;
                sxy += xDiff * yDiff;
                sxx += xDiff * xDiff;
                syy += yDiff * yDiff;
            }

            _correlationScores[j] = (sxx > 1e-10 && syy > 1e-10)
                ? Math.Abs(sxy / Math.Sqrt(sxx * syy))
                : 0;
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _correlationScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeRanks(Vector<T> values, int n)
    {
        var indexed = new (double Value, int Index)[n];
        for (int i = 0; i < n; i++)
            indexed[i] = (NumOps.ToDouble(values[i]), i);

        Array.Sort(indexed, (a, b) => a.Value.CompareTo(b.Value));

        var ranks = new double[n];
        int i2 = 0;
        while (i2 < n)
        {
            int j = i2;
            // Find all equal values
            while (j < n - 1 && Math.Abs(indexed[j + 1].Value - indexed[i2].Value) < 1e-10)
                j++;

            // Assign average rank to all equal values
            double avgRank = (i2 + j) / 2.0 + 1;
            for (int k = i2; k <= j; k++)
                ranks[indexed[k].Index] = avgRank;

            i2 = j + 1;
        }

        return ranks;
    }

    private double[] ComputeFeatureRanks(Matrix<T> data, int featureIdx, int n)
    {
        var indexed = new (double Value, int Index)[n];
        for (int i = 0; i < n; i++)
            indexed[i] = (NumOps.ToDouble(data[i, featureIdx]), i);

        Array.Sort(indexed, (a, b) => a.Value.CompareTo(b.Value));

        var ranks = new double[n];
        int i2 = 0;
        while (i2 < n)
        {
            int j = i2;
            while (j < n - 1 && Math.Abs(indexed[j + 1].Value - indexed[i2].Value) < 1e-10)
                j++;

            double avgRank = (i2 + j) / 2.0 + 1;
            for (int k = i2; k <= j; k++)
                ranks[indexed[k].Index] = avgRank;

            i2 = j + 1;
        }

        return ranks;
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
        if (_selectedIndices is null) return [];

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
