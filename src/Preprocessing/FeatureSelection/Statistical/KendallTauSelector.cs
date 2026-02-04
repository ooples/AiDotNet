using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Statistical;

/// <summary>
/// Kendall's Tau Correlation Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Kendall's tau rank correlation with the target,
/// which is robust to outliers and non-linear monotonic relationships.
/// </para>
/// <para><b>For Beginners:</b> Kendall's tau measures how well the ordering of one
/// variable matches the ordering of another. It counts pairs where both variables
/// agree on which item is "larger" vs pairs where they disagree. It's robust to
/// outliers because it only uses ranks, not actual values.
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
            var x = new double[n];
            for (int i = 0; i < n; i++) x[i] = X[i, j];
            _tauScores[j] = Math.Abs(ComputeKendallTau(x, y, n));
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
        long concordant = 0;
        long discordant = 0;
        long tiedX = 0;
        long tiedY = 0;

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double xDiff = x[i] - x[j];
                double yDiff = y[i] - y[j];

                if (Math.Abs(xDiff) < 1e-10 && Math.Abs(yDiff) < 1e-10)
                    continue; // Both tied
                else if (Math.Abs(xDiff) < 1e-10)
                    tiedX++;
                else if (Math.Abs(yDiff) < 1e-10)
                    tiedY++;
                else if ((xDiff > 0 && yDiff > 0) || (xDiff < 0 && yDiff < 0))
                    concordant++;
                else
                    discordant++;
            }
        }

        long n0 = (long)n * (n - 1) / 2;
        double n1 = n0 - tiedX;
        double n2 = n0 - tiedY;

        if (n1 < 1 || n2 < 1) return 0;

        // Tau-b formula
        return (concordant - discordant) / Math.Sqrt(n1 * n2);
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
