using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.MultiLabel;

/// <summary>
/// Multi-Label ReliefF for multi-label feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Multi-Label ReliefF extends the ReliefF algorithm to handle multi-label
/// classification problems where each instance can belong to multiple classes
/// simultaneously. It considers label correlations when computing feature weights.
/// </para>
/// <para><b>For Beginners:</b> In multi-label problems (like tagging photos with
/// multiple keywords), a single image might be "sunset", "beach", and "romantic"
/// all at once. This algorithm finds features that help distinguish between
/// different label combinations, not just individual labels.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MultiLabelReliefF<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
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
    public int NNeighbors => _nNeighbors;
    public int NIterations => _nIterations;
    public Vector<T>? FeatureWeights => _featureWeights;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MultiLabelReliefF(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int nIterations = 100,
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
            "MultiLabelReliefF requires label matrix. Use Fit(Matrix<T> data, Matrix<T> labels) instead.");
    }

    public void Fit(Matrix<T> data, Matrix<T> labels)
    {
        if (data.Rows != labels.Rows)
            throw new ArgumentException("Number of samples must match between data and labels.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        int numLabels = labels.Columns;

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        var engine = Engine;

        // Binary label vectors for each instance, and how many labels each one has
        var labelVectors = engine.TensorGreaterThan(Tensor<T>.FromMatrix(labels), NumOps.FromDouble(0.5)); // [n, L]
        var labelCounts = engine.ReduceSum(labelVectors, new[] { 1 }, false); // [n]

        // ReliefF differences are scaled by the feature range (Kononenko 1994); a constant feature never differs
        var X = Tensor<T>.FromMatrix(data);
        var range = engine.Reshape(engine.TensorSubtract(FeatureSelectionTensorOps.ColumnMax(X), FeatureSelectionTensorOps.ColumnMin(X)), new[] { 1, p });
        var safeRange = engine.TensorWhere(engine.TensorGreaterThan(range, NumOps.Zero), range, FeatureSelectionTensorOps.Filled(new[] { 1, p }, NumOps.One));
        var scaled = engine.TensorDivide(X, safeRange);

        var weights = new Tensor<T>(new[] { 1, p });
        var everyRow = FeatureSelectionTensorOps.Filled(new[] { n }, NumOps.One);
        T half = NumOps.FromDouble(0.5);

        // Main ReliefF loop
        int numIterations = Math.Min(_nIterations, n);
        for (int iter = 0; iter < numIterations; iter++)
        {
            int idx = random.Next(n);
            var notSelf = engine.TensorSubtract(everyRow, FeatureSelectionTensorOps.OneHot<T>(idx, n));

            // Jaccard label similarity with every other instance
            var own = engine.TensorSlice(labelVectors, new[] { idx, 0 }, new[] { 1, numLabels });
            var intersection = engine.Reshape(engine.TensorMatMul(labelVectors, engine.TensorTranspose(own)), new[] { n });
            var union = engine.TensorSubtract(
                engine.TensorAdd(labelCounts, engine.TensorBroadcastTo(engine.TensorSlice(labelCounts, new[] { idx }, new[] { 1 }), new[] { n })),
                intersection);
            var hasUnion = engine.TensorGreaterThan(union, NumOps.Zero);
            var similarity = engine.TensorWhere(hasUnion,
                engine.TensorDivide(intersection, engine.TensorWhere(hasUnion, union, everyRow)), new Tensor<T>(new[] { n }));

            // Euclidean distances over the range-scaled features (larger closeness = nearer)
            var differences = engine.TensorAbs(engine.TensorSubtract(
                scaled, engine.TensorBroadcastTo(engine.TensorSlice(scaled, new[] { idx, 0 }, new[] { 1, p }), new[] { n, p })));
            var closeness = engine.TensorNegate(engine.TensorSqrt(engine.ReduceSum(engine.TensorSquare(differences), new[] { 1 }, false)));

            // Nearest hits (similar labels) and nearest misses (dissimilar labels), ties by index
            var similar = engine.TensorGreaterThan(similarity, half);
            var hits = FeatureSelectionTensorOps.TopKWhere(closeness, engine.TensorMultiply(notSelf, similar), _nNeighbors);
            var misses = FeatureSelectionTensorOps.TopKWhere(closeness, engine.TensorMultiply(notSelf, FeatureSelectionTensorOps.OneMinus(similar)), _nNeighbors);

            // Update feature weights: each neighbour's difference weighted by (1 - similarity)
            var dissimilarity = FeatureSelectionTensorOps.OneMinus(similarity);
            var hitContrib = NeighbourContribution(differences, dissimilarity, hits, p);
            var missContrib = NeighbourContribution(differences, dissimilarity, misses, p);
            weights = engine.TensorAdd(weights, engine.TensorSubtract(
                engine.TensorDivideScalar(missContrib, FeatureSelectionTensorOps.Count<T>(Math.Max(1, misses.Length))),
                engine.TensorDivideScalar(hitContrib, FeatureSelectionTensorOps.Count<T>(Math.Max(1, hits.Length)))));
        }

        // Normalize weights
        var featureWeights = engine.Reshape(engine.TensorDivideScalar(weights, FeatureSelectionTensorOps.Count<T>(numIterations)), new[] { p });
        _featureWeights = featureWeights.ToVector();

        // Select top features
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(featureWeights, numToSelect);

        IsFitted = true;
    }

    /// <summary>Sum over the neighbours of |difference| * (1 - similarity), per feature ([1, p]).</summary>
    private Tensor<T> NeighbourContribution(Tensor<T> differences, Tensor<T> dissimilarity, int[] neighbours, int p)
    {
        var engine = Engine;
        if (neighbours.Length == 0)
            return new Tensor<T>(new[] { 1, p });

        var index = new Tensor<int>(neighbours, new[] { neighbours.Length });
        var rows = engine.TensorIndexSelect(differences, index, 0); // [k, p]
        var factor = engine.Reshape(engine.TensorIndexSelect(dissimilarity, index, 0), new[] { 1, neighbours.Length });
        return engine.TensorMatMul(factor, rows);
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        // Convert single-label to multi-label format: one indicator column per distinct (rounded) label, ascending
        var classIds = FeatureSelectionInformationOps.ClassIndices(Engine.TensorRound(Tensor<T>.FromVector(target)), out int numLabels);
        var indicators = Engine.TensorOneHot<T>(classIds, numLabels);
        Fit(data, indicators.ToMatrix());
    }

    public Matrix<T> FitTransform(Matrix<T> data, Matrix<T> labels)
    {
        Fit(data, labels);
        return Transform(data);
    }

    public Matrix<T> FitTransform(Matrix<T> data, Vector<T> target)
    {
        Fit(data, target);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelReliefF has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MultiLabelReliefF does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiLabelReliefF has not been fitted.");

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
