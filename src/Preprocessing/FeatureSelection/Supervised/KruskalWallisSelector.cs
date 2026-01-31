using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Supervised;

/// <summary>
/// Kruskal-Wallis H-Test Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Kruskal-Wallis H-test to select features for multi-class problems.
/// This is a non-parametric alternative to one-way ANOVA.
/// </para>
/// <para><b>For Beginners:</b> The Kruskal-Wallis test extends the Wilcoxon test
/// to more than two groups. It checks if different classes have significantly
/// different distributions for a feature, using ranks instead of actual values.
/// </para>
/// </remarks>
public class KruskalWallisSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;

    private double[]? _hStatistics;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double[]? HStatistics => _hStatistics;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public KruskalWallisSelector(
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
            "KruskalWallisSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);
        }

        var classes = y.Distinct().OrderBy(c => c).ToList();
        int k = classes.Count;

        if (k < 2)
            throw new InvalidOperationException("At least 2 classes are required.");

        var classCounts = classes.Select(c => y.Count(yi => yi == c)).ToArray();

        _hStatistics = new double[p];

        for (int j = 0; j < p; j++)
        {
            // Rank all values
            var values = Enumerable.Range(0, n)
                .Select(i => (X[i, j], y[i], i))
                .OrderBy(t => t.Item1)
                .ToList();

            var ranks = new double[n];
            int idx = 0;
            while (idx < n)
            {
                int endIdx = idx;
                while (endIdx < n - 1 && Math.Abs(values[endIdx + 1].Item1 - values[endIdx].Item1) < 1e-10)
                    endIdx++;

                double avgRank = (idx + endIdx) / 2.0 + 1;
                for (int i_r = idx; i_r <= endIdx; i_r++)
                    ranks[values[i_r].i] = avgRank;

                idx = endIdx + 1;
            }

            // Compute rank sums for each class
            var rankSums = new double[k];
            for (int c = 0; c < k; c++)
            {
                int classLabel = classes[c];
                for (int i = 0; i < n; i++)
                    if (y[i] == classLabel)
                        rankSums[c] += ranks[i];
            }

            // Kruskal-Wallis H statistic
            double H = 0;
            for (int c = 0; c < k; c++)
            {
                if (classCounts[c] > 0)
                    H += rankSums[c] * rankSums[c] / classCounts[c];
            }
            H = 12.0 / (n * (n + 1)) * H - 3 * (n + 1);

            // Tie correction (simplified)
            _hStatistics[j] = H;
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _hStatistics[j])
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
            throw new InvalidOperationException("KruskalWallisSelector has not been fitted.");

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
        throw new NotSupportedException("KruskalWallisSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("KruskalWallisSelector has not been fitted.");

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
