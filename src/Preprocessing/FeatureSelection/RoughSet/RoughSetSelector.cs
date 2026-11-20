using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

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
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RoughSetSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _maxFeatures;
    private readonly int _nBins;

    private Vector<T>? _dependencyDegrees;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public Vector<T>? DependencyDegrees => _dependencyDegrees;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
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

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "RoughSetSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize data and round the decisions to integer classes
        var engine = Engine;
        var codes = DiscretizeData(data, n, p);
        var labels = engine.TensorRound(Tensor<T>.FromVector(target));

        // Dependency degree of each feature individually
        var singleCounts = new int[p];
        for (int j = 0; j < p; j++)
            singleCounts[j] = PositiveRegionSize(codes, labels, new[] { j });
        var degrees = engine.TensorDivideScalar(
            new Tensor<int>(singleCounts, new[] { p }).Cast<T>(), NumOps.FromDouble(n));
        _dependencyDegrees = degrees.ToVector();
        var byDependency = PreprocessingTensorOps.StableOrder(degrees, descending: true);

        // Find reduct using greedy algorithm
        var reduct = FindReduct(codes, labels, n, p);

        // If reduct is smaller than maxFeatures, add more by dependency
        if (reduct.Count < _maxFeatures)
        {
            var remaining = byDependency
                .Where(j => !reduct.Contains(j))
                .Take(_maxFeatures - reduct.Count)
                .ToList();
            reduct.UnionWith(remaining);
        }

        // If reduct is larger than maxFeatures, keep top by dependency
        if (reduct.Count > _maxFeatures)
        {
            _selectedIndices = byDependency
                .Where(j => reduct.Contains(j))
                .Take(_maxFeatures)
                .OrderBy(index => index)
                .ToArray();
        }
        else
        {
            _selectedIndices = reduct.OrderBy(index => index).ToArray();
        }

        IsFitted = true;
    }

    /// <summary>Equal-range bin of every value: min(floor((x - min) / range * (nBins - 1)), nBins - 1), 0 for a
    /// column whose range is not above 1e-10.</summary>
    private Tensor<T> DiscretizeData(Matrix<T> data, int n, int p)
    {
        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var min = engine.Reshape(PreprocessingTensorOps.ColumnMin(x), new[] { 1, p });
        var range = engine.TensorSubtract(engine.Reshape(PreprocessingTensorOps.ColumnMax(x), new[] { 1, p }), min);
        var scaled = engine.TensorMultiplyScalar(engine.TensorDivide(engine.TensorSubtract(x, min), range), NumOps.FromDouble(_nBins - 1));
        var bins = engine.TensorMin(engine.TensorFloor(scaled), NumOps.FromDouble(_nBins - 1));
        var varying = engine.TensorBroadcastTo(engine.TensorGreaterThan(range, NumOps.FromDouble(1e-10)), new[] { n, p });
        return SelectorTensorOps.ZeroUnless(varying, bins);
    }

    /// <summary>
    /// Size of the positive region: the rows whose indiscernibility class (equal bins on every feature in
    /// <paramref name="features"/>) holds a single decision.
    /// </summary>
    private int PositiveRegionSize(Tensor<T> codes, Tensor<T> labels, int[] features)
    {
        var engine = Engine;
        int n = codes.Shape[0];

        // Dense class id of each row, refined one feature at a time (ids stay below n * nBins)
        var group = PreprocessingTensorOps.Filled(new[] { n }, NumOps.Zero);
        T radix = NumOps.FromDouble(_nBins);
        int groups = 1;
        foreach (int j in features)
        {
            var key = engine.TensorAdd(engine.TensorMultiplyScalar(group, radix), engine.TensorSliceAxis(codes, 1, j));
            var unique = engine.TensorUniqueWithInfo(key, true, true, false);
            groups = unique.Values.Length;
            group = unique.Inverse is null ? key : unique.Inverse.Cast<T>();
        }

        // A class is consistent when its largest and smallest decisions agree
        var groupIndex = group.Cast<int>();
        var largest = engine.ScatterMax(labels, groupIndex, out _, 0, groups);
        var smallest = engine.TensorNegate(engine.ScatterMax(engine.TensorNegate(labels), groupIndex, out _, 0, groups));
        var consistent = engine.TensorEquals(largest, smallest);
        return engine.TensorCountNonzero(engine.TensorIndexSelect(consistent, groupIndex, 0));
    }

    private HashSet<int> FindReduct(Tensor<T> codes, Tensor<T> labels, int n, int p)
    {
        // Compute full dependency
        var allFeatures = Enumerable.Range(0, p).ToArray();
        double fullDependency = (double)PositiveRegionSize(codes, labels, allFeatures) / n;

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
                double testDependency = (double)PositiveRegionSize(codes, labels, testSet) / n;
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
                double testDependency = (double)PositiveRegionSize(codes, labels, testSet) / n;
                if (Math.Abs(testDependency - currentDependency) < 1e-6)
                    reduct.Remove(reductList[i]);
            }
        }

        return reduct;
    }

    /// <inheritdoc />
    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RoughSetSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RoughSetSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RoughSetSelector has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
