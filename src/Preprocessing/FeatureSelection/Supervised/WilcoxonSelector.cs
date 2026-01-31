using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Wilcoxon Rank-Sum Test Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Wilcoxon rank-sum test (Mann-Whitney U test) to select features
/// that show significant differences between classes without assuming
/// normal distributions.
/// </para>
/// <para><b>For Beginners:</b> The Wilcoxon test is a non-parametric test
/// that checks if one group tends to have larger values than another by
/// comparing ranks. It's robust and doesn't assume your data follows any
/// particular distribution.
/// </para>
/// </remarks>
public class WilcoxonSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _testStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? TestStatistics => _testStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public WilcoxonSelector(
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
            "WilcoxonSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        var y = new int[n];
        for (int i = 0; i < n; i++)
        {
            double yVal = NumOps.ToDouble(target[i]);
            y[i] = yVal > 0.5 ? 1 : 0; // Binarize
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        int n1 = y.Count(yi => yi == 1);
        int n0 = n - n1;

        if (n0 == 0 || n1 == 0)
            throw new InvalidOperationException("Target must have both classes present.");

        _testStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Get values and indices for ranking
            var values = Enumerable.Range(0, n)
                .Select(i => (X[i, j], y[i], i))
                .OrderBy(t => t.Item1)
                .ToList();

            // Assign ranks (handling ties)
            var ranks = new double[n];
            int i_rank = 0;
            while (i_rank < n)
            {
                int j_rank = i_rank;
                while (j_rank < n - 1 && Math.Abs(values[j_rank + 1].Item1 - values[j_rank].Item1) < 1e-10)
                    j_rank++;

                double avgRank = (i_rank + j_rank) / 2.0 + 1;
                for (int k = i_rank; k <= j_rank; k++)
                    ranks[values[k].i] = avgRank;

                i_rank = j_rank + 1;
            }

            // Compute rank sum for class 1
            double R1 = 0;
            for (int i = 0; i < n; i++)
                if (y[i] == 1)
                    R1 += ranks[i];

            // Mann-Whitney U statistic
            double U1 = R1 - n1 * (n1 + 1.0) / 2;
            double U0 = (double)n0 * n1 - U1;
            double U = Math.Min(U1, U0);

            // Standardized statistic (approximate normal for large samples)
            double meanU = n0 * n1 / 2.0;
            double stdU = Math.Sqrt(n0 * n1 * (n0 + n1 + 1.0) / 12);

            _testStatistics[j] = Math.Abs(U - meanU) / (stdU + 1e-10);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _testStatistics[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WilcoxonSelector has not been fitted.");

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
        throw new NotSupportedException("WilcoxonSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("WilcoxonSelector has not been fitted.");

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
