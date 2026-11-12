using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Relief algorithm for feature selection based on instance-based learning.
/// </summary>
/// <remarks>
/// <para>
/// Relief estimates feature quality by sampling instances and computing the
/// difference between distances to nearest hits (same class) and nearest misses
/// (different class). Features that differentiate classes get higher weights.
/// </para>
/// <para><b>For Beginners:</b> Relief works by looking at individual examples.
/// For each example, it finds the nearest example from the same class (hit) and
/// the nearest from a different class (miss). Good features should have similar
/// values for hits and different values for misses. This intuitive approach
/// works well for detecting local feature relevance.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class Relief<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nIterations;
    private readonly int? _randomState;

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

    public Relief(
        int nFeaturesToSelect = 10,
        int nIterations = 100,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nIterations < 1)
            throw new ArgumentException("Number of iterations must be at least 1.", nameof(nIterations));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "Relief requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var labels = FilterStatisticTensorOps.ClassIndex(target, out int classCount);
        var classOneHot = FilterStatisticTensorOps.ClassOneHot<T>(labels, classCount);

        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        int iterations = Math.Min(_nIterations, n);

        // Sample the instances (with replacement), in the same random order as before
        var sampled = Enumerable.Range(0, iterations).Select(_ => rand.Next(n)).ToArray();

        // Manhattan distance from every sampled instance to every instance; the instance itself is no neighbour
        var sampledRows = FeatureSelectionTensorOps.Rows(scaled, sampled);
        var distance = engine.TensorCDist(sampledRows, scaled, 1.0);
        var self = FeatureSelectionTensorOps.SelfMask<T>(sampled, n);
        var sameClass = engine.TensorMatMul(FeatureSelectionTensorOps.Rows(classOneHot, sampled), engine.TensorTranspose(classOneHot));
        T infinity = NumOps.FromDouble(double.PositiveInfinity);

        // nearest hit (same class) and nearest miss (other class), first index on ties
        var (hit, hasHit) = FeatureSelectionTensorOps.FirstArgMinRows(FeatureSelectionTensorOps.KeepOrFill(
            distance, engine.TensorMultiply(sameClass, PreprocessingTensorOps.OneMinus(self)), infinity));
        var (miss, hasMiss) = FeatureSelectionTensorOps.FirstArgMinRows(FeatureSelectionTensorOps.KeepOrFill(
            distance, PreprocessingTensorOps.OneMinus(sameClass), infinity));

        // W += (|x - miss| - |x - hit|) / iterations for the instances that have both
        var diffHit = engine.TensorAbs(engine.TensorSubtract(sampledRows, FeatureSelectionTensorOps.Rows(scaled, hit)));
        var diffMiss = engine.TensorAbs(engine.TensorSubtract(sampledRows, FeatureSelectionTensorOps.Rows(scaled, miss)));
        var both = engine.Reshape(engine.TensorMultiply(hasHit, hasMiss), new[] { 1, iterations });
        var weights = engine.TensorDivideScalar(
            engine.Reshape(engine.TensorMatMul(both, engine.TensorSubtract(diffMiss, diffHit)), new[] { p }),
            NumOps.FromDouble(iterations));
        _featureWeights = weights.ToVector();

        // Select top features
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
            throw new InvalidOperationException("Relief has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("Relief does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("Relief has not been fitted.");

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
