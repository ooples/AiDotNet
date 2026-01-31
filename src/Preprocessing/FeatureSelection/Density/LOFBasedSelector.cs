using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.Density;

/// <summary>
/// Local Outlier Factor (LOF) based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses the Local Outlier Factor concept to select features that best preserve
/// local density structure, helping maintain meaningful neighborhoods.
/// </para>
/// <para><b>For Beginners:</b> LOF measures how "outlying" each point is compared
/// to its neighbors. This selector chooses features that help distinguish normal
/// points from outliers, preserving the local density patterns in your data.
/// </para>
/// </remarks>
public class LOFBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private double[]? _lofContributions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public double[]? LOFContributions => _lofContributions;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public LOFBasedSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 20,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
    }

    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var X = new double[n, p];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < p; j++)
                X[i, j] = NumOps.ToDouble(data[i, j]);

        int k = Math.Min(_nNeighbors, n - 1);

        _lofContributions = new double[p];

        // Compute LOF scores using all features
        var lofAll = ComputeLOF(X, n, p, k);
        double lofVarianceAll = ComputeVariance(lofAll);

        // For each feature, compute LOF without it and measure change
        for (int excludeJ = 0; excludeJ < p; excludeJ++)
        {
            var XReduced = new double[n, p - 1];
            int col = 0;
            for (int j = 0; j < p; j++)
            {
                if (j == excludeJ) continue;
                for (int i = 0; i < n; i++)
                    XReduced[i, col] = X[i, j];
                col++;
            }

            var lofReduced = ComputeLOF(XReduced, n, p - 1, k);
            double lofVarianceReduced = ComputeVariance(lofReduced);

            // Feature importance = how much LOF variance changes when removed
            _lofContributions[excludeJ] = Math.Abs(lofVarianceAll - lofVarianceReduced);
        }

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = Enumerable.Range(0, p)
            .OrderByDescending(j => _lofContributions[j])
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }

    private double[] ComputeLOF(double[,] X, int n, int p, int k)
    {
        // Compute distance matrix
        var distances = new double[n, n];
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double dist = 0;
                for (int f = 0; f < p; f++)
                {
                    double diff = X[i, f] - X[j, f];
                    dist += diff * diff;
                }
                dist = Math.Sqrt(dist);
                distances[i, j] = dist;
                distances[j, i] = dist;
            }
        }

        // Find k-nearest neighbors and k-distance
        var kDistances = new double[n];
        var neighbors = new List<int>[n];
        for (int i = 0; i < n; i++)
        {
            var dists = Enumerable.Range(0, n)
                .Where(j => j != i)
                .OrderBy(j => distances[i, j])
                .ToList();
            neighbors[i] = dists.Take(k).ToList();
            kDistances[i] = distances[i, dists[k - 1]];
        }

        // Compute reachability distances
        double[,] reachDist = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                if (i != j)
                    reachDist[i, j] = Math.Max(kDistances[j], distances[i, j]);

        // Compute local reachability density (LRD)
        var lrd = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            foreach (int j in neighbors[i])
                sum += reachDist[i, j];
            lrd[i] = k / (sum + 1e-10);
        }

        // Compute LOF
        var lof = new double[n];
        for (int i = 0; i < n; i++)
        {
            double sum = 0;
            foreach (int j in neighbors[i])
                sum += lrd[j] / (lrd[i] + 1e-10);
            lof[i] = sum / k;
        }

        return lof;
    }

    private double ComputeVariance(double[] values)
    {
        double mean = values.Average();
        return values.Sum(v => (v - mean) * (v - mean)) / values.Length;
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LOFBasedSelector has not been fitted.");

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
        throw new NotSupportedException("LOFBasedSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LOFBasedSelector has not been fitted.");

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
