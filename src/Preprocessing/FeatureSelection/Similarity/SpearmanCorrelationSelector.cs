using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Similarity;

/// <summary>
/// Spearman Rank Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Spearman rank correlation with the target,
/// measuring monotonic relationships (not necessarily linear).
/// </para>
/// <para><b>For Beginners:</b> Spearman correlation works on ranks rather than raw values.
/// It detects if one variable consistently increases when another increases, even if
/// the relationship isn't a straight line. It's more robust to outliers than Pearson.
/// </para>
/// </remarks>
public class SpearmanCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? Correlations => _correlations;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public SpearmanCorrelationSelector(
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
            "SpearmanCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new double[n];
        for (int i = 0; i < n; i++)
        {
            y[i] = NumOps.ToDouble(target[i]);
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        // Convert y to ranks
        var yRanks = GetRanks(y);

        _correlations = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            var xRanks = GetRanks(col);

            // Compute Pearson correlation on ranks
            double xMean = xRanks.Average();
            double yMean = yRanks.Average();

            double numerator = 0;
            double xSumSq = 0, ySumSq = 0;

            for (int i = 0; i < n; i++)
            {
                double xDiff = xRanks[i] - xMean;
                double yDiff = yRanks[i] - yMean;
                numerator += xDiff * yDiff;
                xSumSq += xDiff * xDiff;
                ySumSq += yDiff * yDiff;
            }

            double denominator = Math.Sqrt(xSumSq * ySumSq);
            _correlations[j] = denominator > 1e-10 ? Math.Abs(numerator / denominator) : 0;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _correlations[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] GetRanks(double[] values)
    {
        int n = values.Length;
        var indexed = values.Select((v, i) => (value: v, index: i))
            .OrderBy(x => x.value)
            .ToArray();

        var ranks = new double[n];
        int i = 0;
        while (i < n)
        {
            int j = i;
            while (j < n - 1 && Math.Abs(indexed[j].value - indexed[j + 1].value) < 1e-10)
                j++;

            double avgRank = (i + j + 2) / 2.0; // Ranks are 1-based
            for (int k = i; k <= j; k++)
                ranks[indexed[k].index] = avgRank;

            i = j + 1;
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
            throw new InvalidOperationException("SpearmanCorrelationSelector has not been fitted.");

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
        throw new NotSupportedException("SpearmanCorrelationSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpearmanCorrelationSelector has not been fitted.");

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
