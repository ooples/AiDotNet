using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Relief for Multi-Label Classification (ReliefMLC).
/// </summary>
/// <remarks>
/// <para>
/// Extends the Relief algorithm to handle multi-label problems. Uses Hamming
/// distance on label sets to determine similarity between instances.
/// </para>
/// <para><b>For Beginners:</b> Standard Relief compares single labels to find
/// hits/misses. ReliefMLC compares label sets, accounting for partial overlap
/// in multi-label scenarios.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ReliefMLC<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;
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

    public ReliefMLC(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
        int nIterations = -1,
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
        _nIterations = nIterations;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "ReliefMLC requires target values. Use Fit(Matrix<T> data, Matrix<T> targets) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> targets)
    {
        if (data.Rows != targets.Rows)
            throw new ArgumentException("Target rows must match data rows.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int nLabels = targets.Columns;
        int m = _nIterations > 0 ? Math.Min(_nIterations, n) : n;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;

        // Normalize features by their range; a feature whose range is below 1e-10 is ignored
        var X = Tensor<T>.FromMatrix(data);
        var range = engine.Reshape(engine.TensorSubtract(PreprocessingTensorOps.ColumnMax(X), PreprocessingTensorOps.ColumnMin(X)), new[] { 1, p });
        var used = PreprocessingTensorOps.OneMinus(engine.TensorLessThan(range, NumOps.FromDouble(1e-10)));
        var safeRange = engine.TensorWhere(used, range, PreprocessingTensorOps.Filled(new[] { 1, p }, NumOps.One));
        var scaled = engine.TensorMultiply(engine.TensorDivide(X, safeRange), engine.TensorBroadcastTo(used, new[] { n, p }));

        // Extract label sets
        var labelSets = engine.TensorGreaterThan(Tensor<T>.FromMatrix(targets), NumOps.FromDouble(0.5)); // [n, L]
        var labelCounts = engine.ReduceSum(labelSets, new[] { 1 }, false); // [n]
        var everyRow = PreprocessingTensorOps.Filled(new[] { n }, NumOps.One);
        int k = Math.Min(_nNeighbors, Math.Max(n - 1, 0));
        var weights = new Tensor<T>(new[] { 1, p });
        T two = NumOps.FromDouble(2);

        // Sample m instances
        var sampleIndices = Enumerable.Range(0, n).OrderBy(_ => random.Next()).Take(m).ToList();

        foreach (int ri in sampleIndices)
        {
            // Find k nearest neighbors based on the range-scaled Euclidean distance (ties by index)
            var differences = engine.TensorAbs(engine.TensorSubtract(
                scaled, engine.TensorBroadcastTo(engine.TensorSlice(scaled, new[] { ri, 0 }, new[] { 1, p }), new[] { n, p })));
            var closeness = engine.TensorNegate(engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(differences), new[] { 1 }, false)));
            var notSelf = engine.TensorSubtract(everyRow, FeatureSelectionTensorOps.OneHot<T>(ri, n));
            var neighbors = FeatureSelectionTensorOps.TopKWhere(closeness, notSelf, k);
            if (neighbors.Length == 0)
                continue;

            // Jaccard label similarity (1 when neither instance has a label)
            var index = new Tensor<int>(neighbors, new[] { neighbors.Length });
            var own = engine.TensorSlice(labelSets, new[] { ri, 0 }, new[] { 1, nLabels });
            var neighbourLabels = engine.TensorIndexSelect(labelSets, index, 0); // [k, L]
            var intersection = engine.Reshape(engine.TensorMatMul(neighbourLabels, engine.TensorTranspose(own)), new[] { neighbors.Length });
            var union = engine.TensorSubtract(
                engine.TensorAdd(engine.TensorIndexSelect(labelCounts, index, 0),
                    engine.TensorBroadcastTo(engine.TensorSlice(labelCounts, new[] { ri }, new[] { 1 }), new[] { neighbors.Length })),
                intersection);
            var hasUnion = engine.TensorGreaterThan(union, NumOps.Zero);
            var neighbourOnes = PreprocessingTensorOps.Filled(new[] { neighbors.Length }, NumOps.One);
            var similarity = engine.TensorWhere(hasUnion, engine.TensorDivide(intersection, engine.TensorWhere(hasUnion, union, neighbourOnes)), neighbourOnes);

            // Similar labels (hit) subtract the difference, different labels (miss) add it:
            // (1 - s) * diff - s * diff = (1 - 2s) * diff
            var factor = engine.TensorSubtract(neighbourOnes, engine.TensorMultiplyScalar(similarity, two));
            weights = engine.TensorAdd(weights, engine.TensorMatMul(
                engine.Reshape(factor, new[] { 1, neighbors.Length }), engine.TensorIndexSelect(differences, index, 0)));
        }

        // Normalize weights
        var featureWeights = engine.Reshape(engine.TensorDivideScalar(weights,
            NumOps.Multiply(FeatureSelectionTensorOps.Count<T>(m), FeatureSelectionTensorOps.Count<T>(_nNeighbors))), new[] { p });
        _featureWeights = featureWeights.ToVector();

        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(featureWeights, nToSelect);

        IsFitted = true;
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> targets)
    {
        Fit(data, targets);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ReliefMLC has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ReliefMLC does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ReliefMLC has not been fitted.");

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
