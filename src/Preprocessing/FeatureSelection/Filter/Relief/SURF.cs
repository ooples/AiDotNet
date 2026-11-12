using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Relief;

/// <summary>
/// Spatially Uniform ReliefF (SURF) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// SURF is a variant of ReliefF that uses distance thresholds instead of k
/// nearest neighbors. It considers all instances within a threshold distance,
/// making it more robust and parameter-free for the neighbor count.
/// </para>
/// <para><b>For Beginners:</b> Unlike ReliefF which looks at exactly k neighbors,
/// SURF looks at all neighbors within a certain distance. This is more natural
/// because it considers the actual distribution of your data - areas with more
/// data points contribute more comparisons.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SURF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _distanceThreshold;

    private Vector<T>? _featureWeights;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double DistanceThreshold => _distanceThreshold;
    public Vector<T>? FeatureWeights => _featureWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SURF(
        int nFeaturesToSelect = 10,
        double distanceThreshold = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (distanceThreshold <= 0)
            throw new ArgumentException("Distance threshold must be positive.", nameof(distanceThreshold));

        _nFeaturesToSelect = nFeaturesToSelect;
        _distanceThreshold = distanceThreshold;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "SURF requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Features scaled to [0, 1] by their range (a constant feature never differs) and integer class labels
        var scaled = FeatureSelectionTensorOps.RangeScaled(FilterStatisticTensorOps.AsTensor(data));
        var classIndex = FilterStatisticTensorOps.ClassIndex(target, out int classCount);
        var classOneHot = FilterStatisticTensorOps.ClassOneHot<T>(classIndex, classCount);
        var distances = engine.TensorCDist(scaled, scaled, 2.0);

        // Threshold: the mean pairwise distance among the first (up to) 100 instances, times the threshold factor
        int sampleSize = Math.Min(100, n);
        int pairCount = sampleSize * (sampleSize - 1) / 2;
        var head = FeatureSelectionTensorOps.Rows(scaled, Enumerable.Range(0, sampleSize).ToArray());
        T avgDist = pairCount > 0
            ? NumOps.Divide(
                NumOps.Multiply(engine.TensorSum(engine.TensorCDist(head, head, 2.0)), NumOps.FromDouble(0.5)),
                NumOps.FromDouble(pairCount))
            : NumOps.One;
        T threshold = NumOps.Multiply(avgDist, NumOps.FromDouble(_distanceThreshold));

        // Neighbours within the threshold, split into hits (same class) and misses; an instance is not its own
        var others = FilterStatisticTensorOps.OneMinus(engine.TensorEye<T>(n));
        var near = engine.TensorMultiply(FilterStatisticTensorOps.OneMinus(engine.TensorGreaterThan(distances, threshold)), others);
        var sameClass = engine.TensorMatMul(classOneHot, engine.TensorTranspose(classOneHot));
        var hits = engine.TensorMultiply(near, sameClass);
        var misses = engine.TensorMultiply(near, FilterStatisticTensorOps.OneMinus(sameClass));

        // W -= diff(hit) / (n * #hits), W += diff(miss) / (n * #misses)
        var weights = FeatureSelectionTensorOps.PairWeightedDifference(
            scaled, FeatureSelectionTensorOps.NearNeighbourPairWeight(hits, misses));
        _featureWeights = weights.ToVector();

        // Select top features
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(weights, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("SURF has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SURF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SURF has not been fitted.");

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
