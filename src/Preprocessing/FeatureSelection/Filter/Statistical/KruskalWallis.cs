using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Kruskal-Wallis H test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Kruskal-Wallis H test is a non-parametric alternative to one-way ANOVA.
/// It tests whether samples from different groups have the same distribution,
/// making it useful for multi-class classification problems.
/// </para>
/// <para><b>For Beginners:</b> When you have more than two groups (like multiple
/// categories), this test checks if the feature values differ across groups.
/// Unlike ANOVA, it doesn't assume the data follows a bell curve, making it
/// more robust for real-world data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KruskalWallis<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _hStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? HStatistics => _hStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KruskalWallis(
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
            "KruskalWallis requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Group samples by class
        var groups = new Dictionary<int, List<int>>();
        for (int i = 0; i < n; i++)
        {
            int classLabel = (int)Math.Round(NumOps.ToDouble(target[i]));
            if (!groups.ContainsKey(classLabel))
                groups[classLabel] = [];
            groups[classLabel].Add(i);
        }

        int numGroups = groups.Count;
        var groupSizes = groups.Values.Select(g => g.Count).ToArray();

        _hStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get all values and compute ranks
            var allValues = new List<(double Value, int Group)>();
            int groupIdx = 0;
            foreach (var group in groups.Values)
            {
                foreach (int i in group)
                    allValues.Add((NumOps.ToDouble(data[i, j]), groupIdx));
                groupIdx++;
            }

            // Sort and assign ranks
            var sorted = allValues.OrderBy(x => x.Value).ToList();
            var ranks = new double[sorted.Count];

            int i2 = 0;
            while (i2 < sorted.Count)
            {
                int start = i2;
                while (i2 < sorted.Count && Math.Abs(sorted[i2].Value - sorted[start].Value) < 1e-10)
                    i2++;

                double avgRank = (start + i2 + 1) / 2.0;
                for (int k = start; k < i2; k++)
                    ranks[k] = avgRank;
            }

            // Compute sum of ranks for each group
            var rankSums = new double[numGroups];
            for (int k = 0; k < sorted.Count; k++)
                rankSums[sorted[k].Group] += ranks[k];

            // Compute H statistic
            double sumSquaredRankSums = 0;
            for (int g = 0; g < numGroups; g++)
                sumSquaredRankSums += rankSums[g] * rankSums[g] / groupSizes[g];

            double H = 12.0 / (n * (n + 1)) * sumSquaredRankSums - 3 * (n + 1);

            // Tie correction
            // (simplified - full correction would count ties)
            _hStatistics[j] = H;

            // P-value from chi-squared distribution approximation
            int df = numGroups - 1;
            _pValues[j] = 1 - ChiSquaredCDF(H, df);
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

    private double ChiSquaredCDF(double x, int df)
    {
        // Approximation using Wilson-Hilferty transformation
        if (x <= 0) return 0;
        if (df <= 0) return 1;

        double z = Math.Pow(x / df, 1.0 / 3) - (1 - 2.0 / (9 * df));
        z /= Math.Sqrt(2.0 / (9 * df));

        return NormalCDF(z);
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
            throw new InvalidOperationException("KruskalWallis has not been fitted.");

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
        throw new NotSupportedException("KruskalWallis does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallis has not been fitted.");

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
