using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Nonparametric;

/// <summary>
/// Kendall's Tau Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on Kendall's tau rank correlation coefficient with
/// the target, which measures ordinal association between features and target.
/// </para>
/// <para><b>For Beginners:</b> Kendall's tau measures how often pairs of data
/// points are in the same order for both feature and target. It's robust to
/// outliers and doesn't assume a linear relationship, just that both variables
/// tend to increase or decrease together.
/// </para>
/// </remarks>
public class KendallTauSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _tauScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TauScores => _tauScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KendallTauSelector(
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
            "KendallTauSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        _tauScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            var col = new double[n];
            for (int i = 0; i < n; i++)
                col[i] = X[i, j];

            _tauScores[j] = Math.Abs(ComputeKendallTau(col, y, n));
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _tauScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeKendallTau(double[] x, double[] y, int n)
    {
        int concordant = 0;
        int discordant = 0;

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double xDiff = x[i] - x[j];
                double yDiff = y[i] - y[j];
                double product = xDiff * yDiff;

                if (product > 0)
                    concordant++;
                else if (product < 0)
                    discordant++;
                // Ties are not counted
            }
        }

        int totalPairs = n * (n - 1) / 2;
        return totalPairs > 0 ? (double)(concordant - discordant) / totalPairs : 0;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTauSelector has not been fitted.");

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
        throw new NotSupportedException("KendallTauSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTauSelector has not been fitted.");

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
