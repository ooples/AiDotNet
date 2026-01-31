using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Wilcoxon Signed-Rank test for paired non-parametric feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Wilcoxon Signed-Rank test compares paired samples without assuming normality.
/// Useful when comparing features across matched/paired observations.
/// </para>
/// <para><b>For Beginners:</b> This test works with paired data (before/after, matched pairs).
/// It ranks the absolute differences and compares positive vs negative ranks.
/// Good for non-normal paired comparisons.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class WilcoxonSignedRank<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _wStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? WStatistics => _wStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WilcoxonSignedRank(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
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

        _wStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Compute differences between feature values and target
            var differences = new List<(double Diff, int Sign)>();
            for (int i = 0; i < n; i++)
            {
                double x = NumOps.ToDouble(data[i, j]);
                double y = NumOps.ToDouble(target[i]);
                double diff = x - y;

                if (Math.Abs(diff) > 1e-10)
                    differences.Add((Math.Abs(diff), diff > 0 ? 1 : -1));
            }

            if (differences.Count == 0)
            {
                _wStatistics[j] = 0;
                _pValues[j] = 1;
                continue;
            }

            // Rank the absolute differences
            var ranked = differences.OrderBy(d => d.Diff).ToList();
            var ranks = new double[ranked.Count];

            int i2 = 0;
            while (i2 < ranked.Count)
            {
                int j2 = i2;
                while (j2 < ranked.Count - 1 && Math.Abs(ranked[j2].Diff - ranked[j2 + 1].Diff) < 1e-10)
                    j2++;

                double avgRank = (i2 + j2) / 2.0 + 1;
                for (int k = i2; k <= j2; k++)
                    ranks[k] = avgRank;

                i2 = j2 + 1;
            }

            // Sum positive and negative ranks
            double wPlus = 0, wMinus = 0;
            for (int k = 0; k < ranked.Count; k++)
            {
                if (ranked[k].Sign > 0)
                    wPlus += ranks[k];
                else
                    wMinus += ranks[k];
            }

            double w = Math.Min(wPlus, wMinus);
            _wStatistics[j] = w;

            // Normal approximation for p-value
            int nPairs = ranked.Count;
            double mu = nPairs * (nPairs + 1) / 4.0;
            double sigma = Math.Sqrt(nPairs * (nPairs + 1) * (2 * nPairs + 1) / 24.0);

            if (sigma > 1e-10)
            {
                double z = Math.Abs((w - mu) / sigma);
                _pValues[j] = 2 * (1 - NormalCDF(z));
            }
            else
            {
                _pValues[j] = 1;
            }
        }

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = _pValues
            .Select((pv, idx) => (PValue: pv, Index: idx))
            .OrderBy(x => x.PValue)
            .Take(nToSelect)
            .Select(x => x.Index)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double NormalCDF(double z)
    {
        return 0.5 * (1 + Erf(z / Math.Sqrt(2)));
    }

    private double Erf(double x)
    {
        double t = 1.0 / (1.0 + 0.5 * Math.Abs(x));
        double tau = t * Math.Exp(-x * x - 1.26551223 +
            t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 +
            t * (1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))));
        return x >= 0 ? 1 - tau : tau - 1;
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
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
