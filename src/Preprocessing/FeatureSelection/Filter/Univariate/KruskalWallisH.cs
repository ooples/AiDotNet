using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Univariate;

/// <summary>
/// Kruskal-Wallis H test for non-parametric multi-class feature selection.
/// </summary>
/// <remarks>
/// <para>
/// The Kruskal-Wallis H test is a non-parametric alternative to one-way ANOVA.
/// It tests whether samples from multiple groups come from the same distribution.
/// </para>
/// <para><b>For Beginners:</b> This test extends Mann-Whitney U to more than 2 groups.
/// It ranks all values together, then checks if different classes have significantly
/// different rank distributions. Works well for non-normal data.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class KruskalWallisH<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
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

    public KruskalWallisH(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "KruskalWallisH requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Get unique classes
        var classMap = new Dictionary<double, int>();
        for (int i = 0; i < n; i++)
        {
            double y = NumOps.ToDouble(target[i]);
            if (!classMap.ContainsKey(y))
                classMap[y] = classMap.Count;
        }

        int k = classMap.Count;
        if (k < 2)
            throw new ArgumentException("KruskalWallisH requires at least 2 classes.");

        _hStatistics = new double[p];
        _pValues = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Group values by class
            var groups = new List<double>[k];
            for (int g = 0; g < k; g++)
                groups[g] = new List<double>();

            var allValues = new List<(double Value, int Group, int OrigIdx)>();
            for (int i = 0; i < n; i++)
            {
                double y = NumOps.ToDouble(target[i]);
                double x = NumOps.ToDouble(data[i, j]);
                int groupIdx = classMap[y];
                groups[groupIdx].Add(x);
                allValues.Add((x, groupIdx, i));
            }

            // Rank all values
            allValues = allValues.OrderBy(v => v.Value).ToList();
            var ranks = new double[n];

            int i2 = 0;
            while (i2 < n)
            {
                int j2 = i2;
                while (j2 < n - 1 && Math.Abs(allValues[j2].Value - allValues[j2 + 1].Value) < 1e-10)
                    j2++;

                double avgRank = (i2 + j2) / 2.0 + 1;
                for (int idx = i2; idx <= j2; idx++)
                    ranks[allValues[idx].OrigIdx] = avgRank;

                i2 = j2 + 1;
            }

            // Compute group rank sums
            var groupRankSums = new double[k];
            var groupCounts = new int[k];
            for (int i3 = 0; i3 < n; i3++)
            {
                double y = NumOps.ToDouble(target[i3]);
                int groupIdx = classMap[y];
                groupRankSums[groupIdx] += ranks[i3];
                groupCounts[groupIdx]++;
            }

            // Compute H statistic
            double h = 0;
            for (int g = 0; g < k; g++)
            {
                if (groupCounts[g] > 0)
                {
                    double meanRank = groupRankSums[g] / groupCounts[g];
                    h += groupCounts[g] * Math.Pow(meanRank - (n + 1) / 2.0, 2);
                }
            }
            h *= 12.0 / (n * (n + 1));

            _hStatistics[j] = h;

            // Chi-square approximation for p-value (df = k - 1)
            _pValues[j] = 1 - ChiSquareCDF(h, k - 1);
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

    private double ChiSquareCDF(double x, int df)
    {
        if (x <= 0) return 0;
        return LowerIncompleteGamma(df / 2.0, x / 2.0) / Gamma(df / 2.0);
    }

    private double LowerIncompleteGamma(double a, double x)
    {
        // Series expansion for lower incomplete gamma
        double sum = 0;
        double term = 1.0 / a;
        sum += term;

        for (int n = 1; n < 100; n++)
        {
            term *= x / (a + n);
            sum += term;
            if (Math.Abs(term) < 1e-10)
                break;
        }

        return Math.Pow(x, a) * Math.Exp(-x) * sum;
    }

    private double Gamma(double x)
    {
        // Lanczos approximation
        double[] g = { 76.18009172947146, -86.50532032941677, 24.01409824083091,
                      -1.231739572450155, 0.1208650973866179e-2, -0.5395239384953e-5 };

        double y = x;
        double tmp = x + 5.5;
        tmp -= (x + 0.5) * Math.Log(tmp);
        double ser = 1.000000000190015;
        for (int j = 0; j < 6; j++)
            ser += g[j] / ++y;

        return Math.Exp(-tmp + Math.Log(2.5066282746310005 * ser / x));
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallisH has not been fitted.");

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
        throw new NotSupportedException("KruskalWallisH does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallisH has not been fitted.");

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
