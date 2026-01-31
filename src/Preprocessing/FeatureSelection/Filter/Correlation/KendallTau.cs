using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Correlation;

/// <summary>
/// Kendall Tau Correlation for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Kendall Tau measures the ordinal association between features and target by
/// counting concordant and discordant pairs. It's robust to outliers and doesn't
/// assume a specific distribution.
/// </para>
/// <para><b>For Beginners:</b> Kendall Tau looks at every pair of data points and
/// asks: "Do they agree on the ranking?" If both feature and target say A > B,
/// that's concordant. If they disagree, that's discordant. More concordant pairs
/// mean stronger positive correlation.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KendallTau<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly bool _useTauB;

    private double[]? _tauScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public bool UseTauB => _useTauB;
    public double[]? TauScores => _tauScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KendallTau(
        int nFeaturesToSelect = 10,
        bool useTauB = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _useTauB = useTauB;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KendallTau requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get target values
        var targetValues = new double[n];
        for (int i = 0; i < n; i++)
            targetValues[i] = NumOps.ToDouble(target[i]);

        _tauScores = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get feature values
            var featureValues = new double[n];
            for (int i = 0; i < n; i++)
                featureValues[i] = NumOps.ToDouble(data[i, j]);

            _tauScores[j] = Math.Abs(ComputeKendallTau(featureValues, targetValues, n));
        }

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _tauScores
            .Select((tau, idx) => (Tau: tau, Index: idx))
            .OrderByDescending(x => x.Tau)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double ComputeKendallTau(double[] x, double[] y, int n)
    {
        int concordant = 0;
        int discordant = 0;
        int tiesX = 0;
        int tiesY = 0;

        for (int i = 0; i < n - 1; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double xDiff = x[i] - x[j];
                double yDiff = y[i] - y[j];

                if (Math.Abs(xDiff) < 1e-10 && Math.Abs(yDiff) < 1e-10)
                {
                    // Tie in both
                    continue;
                }
                else if (Math.Abs(xDiff) < 1e-10)
                {
                    tiesX++;
                }
                else if (Math.Abs(yDiff) < 1e-10)
                {
                    tiesY++;
                }
                else if (xDiff * yDiff > 0)
                {
                    concordant++;
                }
                else
                {
                    discordant++;
                }
            }
        }

        int totalPairs = n * (n - 1) / 2;

        if (_useTauB)
        {
            // Tau-b handles ties
            double denom = Math.Sqrt((totalPairs - tiesX) * (double)(totalPairs - tiesY));
            return denom > 0 ? (concordant - discordant) / denom : 0;
        }
        else
        {
            // Tau-a (simple)
            return totalPairs > 0 ? (concordant - discordant) / (double)totalPairs : 0;
        }
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTau has not been fitted.");

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
        throw new NotSupportedException("KendallTau does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KendallTau has not been fitted.");

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
