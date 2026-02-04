using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Regression;

/// <summary>
/// Mutual Information for regression-based feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Mutual Information Regression estimates the mutual information between each
/// continuous feature and a continuous target. Unlike correlation, MI can capture
/// non-linear dependencies between variables.
/// </para>
/// <para><b>For Beginners:</b> While correlation only finds straight-line relationships,
/// mutual information can detect any pattern where knowing one value tells you about
/// another. This is useful for finding features with curved or complex relationships
/// to your target that linear methods would miss.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class MutualInfoRegression<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
    private readonly int? _randomState;

    private double[]? _miScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? MIScores => _miScores;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public MutualInfoRegression(
        int nFeaturesToSelect = 10,
        int nNeighbors = 3,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nNeighbors < 1)
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
        _randomState = randomState;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MutualInfoRegression requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        _miScores = new double[p];

        // Convert target to double array
        var y = new double[n];
        for (int i = 0; i < n; i++)
            y[i] = NumOps.ToDouble(target[i]);

        for (int j = 0; j < p; j++)
        {
            // Extract feature column
            var x = new double[n];
            for (int i = 0; i < n; i++)
                x[i] = NumOps.ToDouble(data[i, j]);

            // Estimate MI using KNN-based method (Kraskov et al.)
            _miScores[j] = EstimateMI(x, y, n);
        }

        // Select top features by MI
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _miScores[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double EstimateMI(double[] x, double[] y, int n)
    {
        if (n < _nNeighbors + 1) return 0;

        // Normalize data
        var xNorm = Normalize(x, n);
        var yNorm = Normalize(y, n);

        // Compute KNN distances and counts
        var kDistances = new double[n];
        var nxCounts = new int[n];
        var nyCounts = new int[n];

        for (int i = 0; i < n; i++)
        {
            // Find k-th nearest neighbor in joint space
            var jointDists = new List<double>();
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                double dx = Math.Abs(xNorm[i] - xNorm[j]);
                double dy = Math.Abs(yNorm[i] - yNorm[j]);
                jointDists.Add(Math.Max(dx, dy)); // Chebyshev distance
            }
            jointDists.Sort();
            kDistances[i] = jointDists[_nNeighbors - 1];

            // Count points within this distance in marginals
            for (int j = 0; j < n; j++)
            {
                if (i == j) continue;
                if (Math.Abs(xNorm[i] - xNorm[j]) < kDistances[i])
                    nxCounts[i]++;
                if (Math.Abs(yNorm[i] - yNorm[j]) < kDistances[i])
                    nyCounts[i]++;
            }
        }

        // Compute MI estimate using digamma function
        double mi = Digamma(_nNeighbors) + Digamma(n);
        for (int i = 0; i < n; i++)
        {
            mi -= (Digamma(nxCounts[i] + 1) + Digamma(nyCounts[i] + 1)) / n;
        }

        return Math.Max(0, mi);
    }

    private double[] Normalize(double[] values, int n)
    {
        double min = double.MaxValue, max = double.MinValue;
        for (int i = 0; i < n; i++)
        {
            if (values[i] < min) min = values[i];
            if (values[i] > max) max = values[i];
        }

        var result = new double[n];
        double range = max - min;
        if (range < 1e-10)
        {
            for (int i = 0; i < n; i++)
                result[i] = 0.5;
        }
        else
        {
            for (int i = 0; i < n; i++)
                result[i] = (values[i] - min) / range;
        }

        return result;
    }

    private double Digamma(double x)
    {
        if (x <= 0) return double.NegativeInfinity;

        // Asymptotic expansion for large x
        if (x > 8)
        {
            double inv = 1.0 / x;
            double inv2 = inv * inv;
            return Math.Log(x) - 0.5 * inv - inv2 / 12 + inv2 * inv2 / 120;
        }

        // Recurrence relation for small x
        double result = 0;
        while (x < 8)
        {
            result -= 1.0 / x;
            x += 1;
        }

        double inv_x = 1.0 / x;
        double inv2_x = inv_x * inv_x;
        result += Math.Log(x) - 0.5 * inv_x - inv2_x / 12 + inv2_x * inv2_x / 120;

        return result;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInfoRegression has not been fitted.");

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
        throw new NotSupportedException("MutualInfoRegression does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MutualInfoRegression has not been fitted.");

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
