using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Texture;

/// <summary>
/// Local Binary Pattern inspired Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their local pattern characteristics computed
/// from neighboring value comparisons.
/// </para>
/// <para><b>For Beginners:</b> This looks at how each value compares to its neighbors,
/// creating a "pattern" at each point. Features with more interesting local patterns
/// often carry more discriminative information.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LocalBinaryPatternSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _neighborhoodSize;

    private Vector<T>? _patternScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NeighborhoodSize => _neighborhoodSize;
    public Vector<T>? PatternScores => _patternScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LocalBinaryPatternSelector(
        int nFeaturesToSelect = 10,
        int neighborhoodSize = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _neighborhoodSize = neighborhoodSize;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        int halfNeighbor = _neighborhoodSize / 2;
        int total = n - 2 * halfNeighbor;
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (total > 0 && p > 0)
        {
            // Local binary pattern code of every interior row: bit b is set when neighbour k (k != 0) >= centre
            var center = engine.TensorSlice(x, new[] { halfNeighbor, 0 }, new[] { total, p });
            // (Horner from the highest bit, k = +halfNeighbor, down to bit 0, k = -halfNeighbor)
            var codes = PreprocessingTensorOps.Filled(new[] { total, p }, NumOps.Zero);
            T two = NumOps.FromDouble(2);
            int bit = 0;
            for (int k = halfNeighbor; k >= -halfNeighbor; k--)
            {
                if (k == 0) continue;
                var neighbor = engine.TensorSlice(x, new[] { halfNeighbor + k, 0 }, new[] { total, p });
                var isSet = PreprocessingTensorOps.Not(engine.TensorLessThan(neighbor, center));
                codes = engine.TensorAdd(engine.TensorMultiplyScalar(codes, two), isSet);
                bit++;
            }

            // Pattern counts per column: offset each column's codes into its own range, then count equal keys
            var columnOffset = engine.TensorMultiplyScalar(
                engine.Reshape(engine.TensorArange<T>(0, p, 1), new[] { 1, p }), NumOps.FromDouble(Math.Pow(2, bit)));
            var keys = engine.Reshape(engine.TensorAdd(codes, columnOffset), new[] { total * p });
            var sortedKeys = engine.TensorSort(keys, 0, false).Values;
            var counts = engine.TensorSubtract(
                engine.TensorSearchSorted(sortedKeys, keys, true),
                engine.TensorSearchSorted(sortedKeys, keys, false)).Cast<T>();

            // Entropy (bits) of the pattern distribution: -sum over rows of (1/total) * log2(count / total)
            var probability = engine.TensorDivideScalar(engine.Reshape(counts, new[] { total, p }), NumOps.FromDouble(total));
            scores = engine.TensorNegate(engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorLog2(probability), new[] { 0 }, false), NumOps.FromDouble(total)));
        }

        _patternScores = scores.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LocalBinaryPatternSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LocalBinaryPatternSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LocalBinaryPatternSelector has not been fitted.");

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
