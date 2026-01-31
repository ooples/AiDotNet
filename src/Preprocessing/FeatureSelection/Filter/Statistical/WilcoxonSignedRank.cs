using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Wilcoxon Signed-Rank test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Wilcoxon Signed-Rank test is a non-parametric test for paired samples.
/// It tests whether the median difference between pairs of observations is zero,
/// useful for before/after comparisons or matched samples.
/// </para>
/// <para><b>For Beginners:</b> When you have paired data (like measurements from
/// the same subjects at different times), this test checks if there's a consistent
/// difference. It doesn't assume normal distributions, making it more robust than
/// paired t-tests for real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class WilcoxonSignedRank<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private double[]? _wStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double[]? WStatistics => _wStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WilcoxonSignedRank(
        int nFeaturesToSelect = 10,
        double alpha = 0.05,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _alpha = alpha;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "WilcoxonSignedRank requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // For feature selection, compare each feature to the target
        _wStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute differences between feature and target
            var differences = new List<(double AbsDiff, int Sign)>();

            for (int i = 0; i < n; i++)
            {
                double diff = NumOps.ToDouble(data[i, j]) - NumOps.ToDouble(target[i]);
                if (Math.Abs(diff) > 1e-10) // Exclude zero differences
                {
                    differences.Add((Math.Abs(diff), Math.Sign(diff)));
                }
            }

            if (differences.Count < 2)
            {
                _wStatistics[j] = 0;
                _pValues[j] = 1.0;
                continue;
            }

            // Rank the absolute differences
            var ranked = differences
                .OrderBy(d => d.AbsDiff)
                .Select((d, idx) => (d.AbsDiff, d.Sign, Rank: idx + 1.0))
                .ToList();

            // Handle ties (average ranks)
            int i2 = 0;
            while (i2 < ranked.Count)
            {
                int start = i2;
                while (i2 < ranked.Count && Math.Abs(ranked[i2].AbsDiff - ranked[start].AbsDiff) < 1e-10)
                    i2++;

                double avgRank = (start + i2 + 1) / 2.0;
                for (int k = start; k < i2; k++)
                    ranked[k] = (ranked[k].AbsDiff, ranked[k].Sign, avgRank);
            }

            // Compute W+ (sum of positive ranks) and W- (sum of negative ranks)
            double wPlus = ranked.Where(r => r.Sign > 0).Sum(r => r.Rank);
            double wMinus = ranked.Where(r => r.Sign < 0).Sum(r => r.Rank);

            double W = Math.Min(wPlus, wMinus);
            _wStatistics[j] = W;

            // Normal approximation for p-value
            int nPairs = differences.Count;
            double mean = nPairs * (nPairs + 1) / 4.0;
            double stdDev = Math.Sqrt(nPairs * (nPairs + 1) * (2 * nPairs + 1) / 24.0);

            if (stdDev > 0)
            {
                double z = (W - mean) / stdDev;
                _pValues[j] = 2 * NormalCDF(-Math.Abs(z));
            }
            else
            {
                _pValues[j] = 1.0;
            }
        }

        // Select features with smallest p-values
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _pValues
            .Select((pval, idx) => (PValue: pval, Index: idx))
            .OrderBy(x => x.PValue)
            .Take(numToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double NormalCDF(double x)
    {
        double t = 1.0 / (1.0 + 0.2316419 * Math.Abs(x));
        double d = 0.3989423 * Math.Exp(-x * x / 2);
        double p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
        return x > 0 ? 1 - p : p;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WilcoxonSignedRank has not been fitted.");

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
        throw new NotSupportedException("WilcoxonSignedRank does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WilcoxonSignedRank has not been fitted.");

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
