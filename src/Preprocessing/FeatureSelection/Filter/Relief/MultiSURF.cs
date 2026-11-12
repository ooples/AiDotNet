using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// MultiSURF algorithm with adaptive distance thresholds per instance.
/// </summary>
/// <remarks>
/// <para>
/// MultiSURF improves SURF by using instance-specific distance thresholds
/// based on the local density around each instance. This adapts to varying
/// data densities across the feature space.
/// </para>
/// <para><b>For Beginners:</b> While SURF uses one global distance threshold,
/// MultiSURF calculates a custom threshold for each sample based on how
/// spread out its neighbors are. This works better for datasets with
/// varying density regions.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MultiSURF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureWeights => _featureWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MultiSURF(int nFeaturesToSelect = 10, int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MultiSURF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;

        // Features scaled to [0, 1] by their range; a constant feature never differs between instances
        var scaled = FeatureSelectionTensorOps.RangeScaled(FilterStatisticTensorOps.AsTensor(data));
        var classOneHot = StatisticalTestHelper.ClassOneHot(target, out _);

        // All pairwise distances; an instance is not its own neighbour
        var distances = engine.TensorCDist(scaled, scaled, 2.0);
        var others = PreprocessingTensorOps.OneMinus(engine.TensorEye<T>(n));
        T otherCount = NumOps.FromDouble(n - 1);

        // Per-instance threshold: mean - 0.5 * std of its distances to the others (closer than average)
        var meanDist = engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorMultiply(distances, others), new[] { 1 }, false), otherCount);
        var deviation = engine.TensorMultiply(
            engine.TensorSubtract(distances, FilterStatisticTensorOps.ColumnBroadcast(meanDist, n)), others);
        var stdDist = engine.TensorSqrt(engine.TensorDivideScalar(
            engine.ReduceSum(engine.TensorSquare(deviation), new[] { 1 }, false), otherCount));
        var threshold = engine.TensorSubtract(meanDist, engine.TensorMultiplyScalar(stdDist, NumOps.FromDouble(0.5)));

        // Near neighbours (within the threshold), split into hits (same class) and misses
        var near = engine.TensorMultiply(PreprocessingTensorOps.OneMinus(engine.TensorGreaterThan(
            distances, FilterStatisticTensorOps.ColumnBroadcast(threshold, n))), others);
        var sameClass = engine.TensorMatMul(classOneHot, engine.TensorTranspose(classOneHot));
        var hits = engine.TensorMultiply(near, sameClass);
        var misses = engine.TensorMultiply(near, PreprocessingTensorOps.OneMinus(sameClass));

        // W -= diff(hit) / (n * #hits), W += diff(miss) / (n * #misses)
        var weights = FeatureSelectionTensorOps.PairWeightedDifference(
            scaled, FeatureSelectionTensorOps.NearNeighbourPairWeight(hits, misses));
        _featureWeights = weights.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(weights, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
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
            throw new InvalidOperationException("MultiSURF has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MultiSURF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiSURF has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
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
