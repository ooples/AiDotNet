using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Causal;

/// <summary>
/// Markov Blanket Feature Selection for finding causal features.
/// </summary>
/// <remarks>
/// <para>
/// Identifies the Markov blanket of the target variable - the minimal set of
/// features that makes the target conditionally independent of all other
/// features. These features are the most relevant for prediction.
/// </para>
/// <para><b>For Beginners:</b> Imagine the target variable has a "blanket" of
/// features around it that shields it from all other features. If you know
/// the blanket features, knowing any other feature doesn't help predict the
/// target. This method finds that protective blanket of features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MarkovBlanketSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>
{
    private readonly int _maxFeatures;
    private readonly double _threshold;
    private readonly int _nBins;

    private Vector<T>? _relevanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    public int MaxFeatures => _maxFeatures;
    public Vector<T>? RelevanceScores => _relevanceScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MarkovBlanketSelector(
        int maxFeatures = 20,
        double threshold = 0.01,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (maxFeatures < 1)
            throw new ArgumentException("Max features must be at least 1.", nameof(maxFeatures));
        if (nBins < 1)
            throw new ArgumentException("nBins must be at least 1.", nameof(nBins));

        _maxFeatures = maxFeatures;
        _threshold = threshold;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MarkovBlanketSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize data (one-hot bins per feature) and the target
        var engine = Engine;
        var xBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);
        var yBins = PreprocessingTensorOps.RangeScaledBinOneHot(engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins);

        // Mutual information with target for each feature
        var relevance = engine.Reshape(
            PreprocessingTensorOps.HistogramMutualInformation(xBins, yBins, _nBins, bits: false, logEpsilon: 0, clampAtZero: false),
            new[] { p });
        _relevanceScores = relevance.ToVector();

        // Grow-Shrink algorithm for Markov blanket
        var blanket = new HashSet<int>();
        var candidates = PreprocessingTensorOps.StableOrder(relevance, descending: true);
        T threshold = NumOps.FromDouble(_threshold);

        // Growing phase: add features that increase conditional MI
        foreach (int candidate in candidates)
        {
            if (blanket.Count >= _maxFeatures) break;

            if (PreprocessingTensorOps.IsAbove(ConditionalMutualInformation(xBins, yBins, candidate, blanket, n), threshold))
                blanket.Add(candidate);
        }

        // Shrinking phase: remove features that become redundant
        var blanketList = blanket.ToList();
        for (int i = blanketList.Count - 1; i >= 0; i--)
        {
            int feature = blanketList[i];
            var others = blanket.Where(f => f != feature).ToHashSet();

            if (!PreprocessingTensorOps.IsAbove(ConditionalMutualInformation(xBins, yBins, feature, others, n), threshold))
                blanket.Remove(feature);
        }

        _selectedIndices = blanket.OrderBy(x => x).ToArray();

        // If blanket is empty, select top features by MI
        if (_selectedIndices.Length == 0)
            _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(relevance, Math.Min(Math.Min(5, p), _maxFeatures));

        IsFitted = true;
    }

    private Tensor<T> FeatureBins(Tensor<T> bins, int feature) =>
        Engine.TensorSlice(bins, new[] { 0, feature * _nBins }, new[] { bins.Shape[0], _nBins });

    private T ConditionalMutualInformation(Tensor<T> xBins, Tensor<T> yBins, int feature, HashSet<int> conditionSet, int n)
    {
        var engine = Engine;
        var featureBins = FeatureBins(xBins, feature);
        var unconditioned = engine.Reshape(
            PreprocessingTensorOps.HistogramMutualInformation(featureBins, yBins, _nBins, bits: false, logEpsilon: 0, clampAtZero: false),
            new[] { 1 });
        if (conditionSet.Count == 0)
            return engine.TensorSum(unconditioned);

        // MI conditioned on each conditioning variable, averaged over the variables that have a group of more than 5 rows
        Tensor<T>? total = null;
        int condCount = 0;
        foreach (int condFeature in conditionSet)
        {
            var groupMI = GroupedMutualInformation(FeatureBins(xBins, condFeature), featureBins, yBins, n, out bool anyGroup);
            if (!anyGroup)
                continue;

            total = total is null ? groupMI : engine.TensorAdd(total, groupMI);
            condCount++;
        }

        return total is null
            ? engine.TensorSum(unconditioned)
            : NumOps.Divide(engine.TensorSum(total), NumOps.FromDouble(condCount));
    }

    /// <summary>
    /// The sum over groups g with more than 5 rows of MI_g(x; y) * |g| / n, where the groups are the values of
    /// the conditioning variable and MI_g uses the probabilities within the group. Returns a [1] tensor.
    /// </summary>
    private Tensor<T> GroupedMutualInformation(Tensor<T> groupBins, Tensor<T> featureBins, Tensor<T> yBins, int n, out bool anyGroup)
    {
        var engine = Engine;
        int b = _nBins;
        var cube = new[] { b, b, b };

        // joint[g, x, y] counts from the row-wise outer product of the group and feature one-hots
        var groupFeature = engine.TensorMultiply(
            engine.TensorBroadcastTo(engine.Reshape(groupBins, new[] { n, b, 1 }), new[] { n, b, b }),
            engine.TensorBroadcastTo(engine.Reshape(featureBins, new[] { n, 1, b }), new[] { n, b, b }));
        var joint = engine.Reshape(
            engine.TensorMatMul(engine.TensorTranspose(engine.Reshape(groupFeature, new[] { n, b * b })), yBins), cube);

        var groupSize = engine.ReduceSum(joint, new[] { 1, 2 }, true);   // [b, 1, 1]
        var xCounts = engine.ReduceSum(joint, new[] { 2 }, true);        // [b, b, 1]
        var yCounts = engine.ReduceSum(joint, new[] { 1 }, true);        // [b, 1, b]
        var size = engine.TensorBroadcastTo(groupSize, cube);
        var pxy = engine.TensorDivide(joint, size);
        var px = engine.TensorDivide(engine.TensorBroadcastTo(xCounts, cube), size);
        var py = engine.TensorDivide(engine.TensorBroadcastTo(yCounts, cube), size);
        var terms = engine.TensorMultiply(pxy, engine.TensorLog(engine.TensorDivide(pxy, engine.TensorMultiply(px, py))));
        var cells = engine.TensorWhere(engine.TensorGreaterThan(joint, NumOps.Zero), terms, new Tensor<T>(cube));
        var miPerGroup = engine.ReduceSum(cells, new[] { 1, 2 }, false);  // [b]

        var sizes = engine.Reshape(groupSize, new[] { b });
        var large = engine.TensorGreaterThan(sizes, NumOps.FromDouble(5));
        anyGroup = PreprocessingTensorOps.CountNonZero(large) > 0;
        var weighted = engine.TensorMultiply(engine.TensorMultiply(miPerGroup, large), engine.TensorDivideScalar(sizes, NumOps.FromDouble(n)));
        return engine.ReduceSum(engine.Reshape(weighted, new[] { 1, b }), new[] { 1 }, false);
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
            throw new InvalidOperationException("MarkovBlanketSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MarkovBlanketSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MarkovBlanketSelector has not been fitted.");

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
