using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Statistical;

/// <summary>
/// Mann-Whitney U test for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Mann-Whitney U test is a non-parametric test that compares the distributions
/// of a feature between two groups. It doesn't assume normality, making it robust
/// for real-world data with outliers or skewed distributions.
/// </para>
/// <para><b>For Beginners:</b> This test asks: "Do the values of this feature tend to
/// be higher in one group than another?" Unlike t-tests, it doesn't require the data
/// to follow a bell curve. It's great for data that's skewed or has outliers.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MannWhitneyU<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _alpha;

    private double[]? _uStatistics;
    private double[]? _pValues;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double Alpha => _alpha;
    public double[]? UStatistics => _uStatistics;
    public double[]? PValues => _pValues;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MannWhitneyU(
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
            "MannWhitneyU requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Split into two groups based on target (binary classification)
        var group0 = new List<int>();
        var group1 = new List<int>();

        double threshold = 0.5;
        for (int i = 0; i < n; i++)
        {
            if (NumOps.ToDouble(target[i]) < threshold)
                group0.Add(i);
            else
                group1.Add(i);
        }

        int n0 = group0.Count;
        int n1 = group1.Count;

        if (n0 < 1 || n1 < 1)
        {
            // Fallback if only one class
            _uStatistics = new double[p];
            _pValues = Enumerable.Repeat(1.0, p).ToArray();
            _selectedIndices = Enumerable.Range(0, Math.Min(_nFeaturesToSelect, p)).ToArray();
            IsFitted = true;
            return;
        }

        _uStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get values for each group
            var values0 = group0.Select(i => NumOps.ToDouble(data[i, j])).ToArray();
            var values1 = group1.Select(i => NumOps.ToDouble(data[i, j])).ToArray();

            // Compute U statistic
            double u = ComputeU(values0, values1);
            _uStatistics[j] = u;

            // Compute p-value using normal approximation
            double mu = (double)n0 * n1 / 2;
            double sigma = Math.Sqrt((double)n0 * n1 * (n0 + n1 + 1) / 12);

            double z = (u - mu) / sigma;
            _pValues[j] = 2 * (1 - NormalCDF(Math.Abs(z)));
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

    private double ComputeU(double[] group0, double[] group1)
    {
        int n0 = group0.Length;
        int n1 = group1.Length;

        // Combine and rank all values
        var combined = group0.Select(v => (Value: v, Group: 0))
            .Concat(group1.Select(v => (Value: v, Group: 1)))
            .OrderBy(x => x.Value)
            .ToArray();

        // Assign ranks (handling ties with average rank)
        var ranks = new double[combined.Length];
        int i = 0;
        while (i < combined.Length)
        {
            int start = i;
            while (i < combined.Length && Math.Abs(combined[i].Value - combined[start].Value) < 1e-10)
                i++;

            double avgRank = (start + i + 1) / 2.0;
            for (int k = start; k < i; k++)
                ranks[k] = avgRank;
        }

        // Sum ranks for group 0
        double r0 = 0;
        for (int k = 0; k < combined.Length; k++)
        {
            if (combined[k].Group == 0)
                r0 += ranks[k];
        }

        // U statistic for group 0
        double u0 = r0 - (double)n0 * (n0 + 1) / 2;

        // Return smaller U
        double u1 = (double)n0 * n1 - u0;
        return Math.Min(u0, u1);
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
            throw new InvalidOperationException("MannWhitneyU has not been fitted.");

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
        throw new NotSupportedException("MannWhitneyU does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MannWhitneyU has not been fitted.");

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
