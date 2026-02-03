using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Preprocessing.FeatureSelection.RoughSet;

/// <summary>
/// Rough Set Theory-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses rough set theory concepts like positive region and dependency degree
/// to find the minimal subset of features (reduct) that preserves the
/// classification ability of the full feature set.
/// </para>
/// <para><b>For Beginners:</b> Rough set theory handles uncertainty by grouping
/// similar objects together. If removing a feature doesn't change which groups
/// are distinguishable, that feature is redundant. This method finds the
/// smallest set of features that still tells all objects apart.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
public class RoughSetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _maxFeatures;
    private readonly int _nBins;

    private double[]? _dependencyDegrees;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public double[]? DependencyDegrees => _dependencyDegrees;
    public int[]? SelectedIndices => _selectedIndices;
    public override bool SupportsInverseTransform => false;

    public RoughSetSelector(
        int maxFeatures = 20,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));

        _maxFeatures = maxFeatures;
        _nBins = nBins;
    }

    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RoughSetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize data
        var X = DiscretizeData(data, n, p);
        var y = new int[n];
        for (int i = 0; i < n; i++)
            y[i] = (int)Math.Round(NumOps.ToDouble(target[i]));

        // Compute dependency degree for each feature individually
        _dependencyDegrees = new double[p];
        for (int j = 0; j < p; j++)
            _dependencyDegrees[j] = ComputeDependencyDegree(X, y, new[] { j }, n, p);

        // Find reduct using greedy algorithm
        var reduct = FindReduct(X, y, n, p);

        // If reduct is smaller than maxFeatures, add more by dependency
        if (reduct.Count < _maxFeatures)
        {
            var remaining = Enumerable.Range(0, p)
                .Except(reduct)
                .OrderByDescending(j => _dependencyDegrees[j])
                .Take(_maxFeatures - reduct.Count);
            reduct.UnionWith(remaining);
        }

        // If reduct is larger than maxFeatures, keep top by dependency
        if (reduct.Count > _maxFeatures)
        {
            _selectedIndices = reduct
                .OrderByDescending(j => _dependencyDegrees[j])
                .Take(_maxFeatures)
                .OrderBy(x => x)
                .ToArray();
        }
        else
        {
            _selectedIndices = reduct.OrderBy(x => x).ToArray();
        }

        IsFitted = true;
    }

    private int[,] DiscretizeData(Matrix<T> data, int n, int p)
    {
        var result = new int[n, p];
        for (int j = 0; j < p; j++)
        {
            double min = double.MaxValue, max = double.MinValue;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                min = Math.Min(min, val);
                max = Math.Max(max, val);
            }

            double range = max - min;
            for (int i = 0; i < n; i++)
            {
                double val = NumOps.ToDouble(data[i, j]);
                result[i, j] = range > 1e-10
                    ? Math.Min((int)((val - min) / range * (_nBins - 1)), _nBins - 1)
                    : 0;
            }
        }
        return result;
    }

    private double ComputeDependencyDegree(int[,] X, int[] y, int[] features, int n, int p)
    {
        // Compute indiscernibility relation based on selected features
        var equivalenceClasses = new Dictionary<string, List<int>>();
        for (int i = 0; i < n; i++)
        {
            var key = string.Join(",", features.Select(j => X[i, j]));
            if (!equivalenceClasses.ContainsKey(key))
                equivalenceClasses[key] = new List<int>();
            equivalenceClasses[key].Add(i);
        }

        // Compute positive region
        int positiveRegionSize = 0;
        foreach (var eqClass in equivalenceClasses.Values)
        {
            // Check if all objects in equivalence class have same decision
            var decisions = eqClass.Select(i => y[i]).Distinct().ToList();
            if (decisions.Count == 1)
                positiveRegionSize += eqClass.Count;
        }

        return (double)positiveRegionSize / n;
    }

    private HashSet<int> FindReduct(int[,] X, int[] y, int n, int p)
    {
        // Compute full dependency
        var allFeatures = Enumerable.Range(0, p).ToArray();
        double fullDependency = ComputeDependencyDegree(X, y, allFeatures, n, p);

        // Greedy forward selection
        var reduct = new HashSet<int>();
        double currentDependency = 0;

        while (currentDependency < fullDependency - 1e-6 && reduct.Count < p)
        {
            int bestFeature = -1;
            double bestIncrease = 0;

            for (int j = 0; j < p; j++)
            {
                if (reduct.Contains(j)) continue;

                var testSet = reduct.Union(new[] { j }).ToArray();
                double testDependency = ComputeDependencyDegree(X, y, testSet, n, p);
                double increase = testDependency - currentDependency;

                if (increase > bestIncrease)
                {
                    bestIncrease = increase;
                    bestFeature = j;
                }
            }

            if (bestFeature < 0 || bestIncrease < 1e-6)
                break;

            reduct.Add(bestFeature);
            currentDependency += bestIncrease;
        }

        // Try to remove redundant features
        var reductList = reduct.ToList();
        for (int i = reductList.Count - 1; i >= 0; i--)
        {
            var testSet = reduct.Except(new[] { reductList[i] }).ToArray();
            if (testSet.Length > 0)
            {
                double testDependency = ComputeDependencyDegree(X, y, testSet, n, p);
                if (Math.Abs(testDependency - currentDependency) < 1e-6)
                    reduct.Remove(reductList[i]);
            }
        }

        return reduct;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RoughSetSelector has not been fitted.");

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
        throw new NotSupportedException("RoughSetSelector does not support inverse transformation.");
    }

    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RoughSetSelector has not been fitted.");

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
