using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Spearman Rank Correlation Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Spearman rank correlation with the target,
/// which measures monotonic relationships without assuming linearity.
/// </para>
/// <para><b>For Beginners:</b> Spearman correlation uses ranks instead of actual
/// values. It converts each variable to ranks (1st, 2nd, 3rd, etc.) and then
/// computes the correlation of those ranks. This makes it robust to outliers
/// and able to detect non-linear but monotonic (always increasing or decreasing)
/// relationships.
/// </para>
/// </remarks>
public class SpearmanCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _correlationScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? CorrelationScores => _correlationScores;
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

        // Rank the target
        var yRanks = ComputeRanks(y, n);

        _correlationScores = new double[p];
        for (int j = 0; j < p; j++)
        {
            var x = new double[n];
            for (int i = 0; i < n; i++) x[i] = X[i, j];
            var xRanks = ComputeRanks(x, n);

            // Pearson correlation of ranks
            _correlationScores[j] = Math.Abs(ComputePearsonCorrelation(xRanks, yRanks, n));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _correlationScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeRanks(double[] values, int n)
    {
        var indexed = values.Select((v, i) => (v, i)).OrderBy(t => t.v).ToList();
        var ranks = new double[n];

        int i = 0;
        while (i < n)
        {
            int j = i;
            while (j < n - 1 && Math.Abs(indexed[j + 1].v - indexed[j].v) < 1e-10)
                j++;

            // Average rank for ties
            double avgRank = (i + j) / 2.0 + 1;
            for (int k = i; k <= j; k++)
                ranks[indexed[k].i] = avgRank;

            i = j + 1;
        }

        return ranks;
    }

    private double ComputePearsonCorrelation(double[] x, double[] y, int n)
    {
        double xMean = x.Average();
        double yMean = y.Average();

        double sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < n; i++)
        {
            double xd = x[i] - xMean;
            double yd = y[i] - yMean;
            sxy += xd * yd;
            sxx += xd * xd;
            syy += yd * yd;
        }

        return (sxx > 1e-10 && syy > 1e-10) ? sxy / Math.Sqrt(sxx * syy) : 0;
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
