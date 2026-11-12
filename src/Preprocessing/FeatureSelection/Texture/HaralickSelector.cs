using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Texture;

/// <summary>
/// Haralick Texture Features based Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their texture characteristics computed from
/// co-occurrence statistics, useful for image-derived or spatial data features.
/// </para>
/// <para><b>For Beginners:</b> Haralick features describe patterns like contrast,
/// homogeneity, and correlation in data. This selector identifies features
/// with strong textural patterns that often indicate meaningful structure.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class HaralickSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nLevels;

    private Vector<T>? _textureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NLevels => _nLevels;
    public Vector<T>? TextureScores => _textureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public HaralickSelector(
        int nFeaturesToSelect = 10,
        int nLevels = 16,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nLevels = nLevels;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        var scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (n > 0 && p > 0)
        {
            // Entropy (bits) of the equal-width level histogram; constant columns contribute 0
            var levels = PreprocessingTensorOps.EqualWidthBinOneHot(x, _nLevels, out var varying);
            var counts = engine.TensorTranspose(
                engine.Reshape(engine.ReduceSum(levels, new[] { 0 }, false), new[] { p, _nLevels }));
            var entropy = engine.TensorMultiply(SelectorTensorOps.ColumnEntropyBits(counts, 0, 0), varying);
            scores = entropy;

            if (n >= 2)
            {
                // Contrast (mean squared step) and homogeneity (mean 1 / (1 + |step|)) between consecutive rows
                var steps = engine.TensorDiff(x, 1, 0);
                var contrast = engine.TensorDivideScalar(
                    SelectorTensorOps.ColumnSumOfSquares(steps), NumOps.FromDouble(n - 1));
                var similarity = engine.TensorDivide(
                    PreprocessingTensorOps.Filled(new[] { n - 1, p }, NumOps.One),
                    engine.TensorAddScalar(engine.TensorAbs(steps), NumOps.One));
                var homogeneity = engine.TensorDivideScalar(
                    engine.ReduceSum(similarity, new[] { 0 }, false), NumOps.FromDouble(n - 1));
                scores = engine.TensorAdd(engine.TensorAdd(contrast, homogeneity), entropy);
            }
        }

        _textureScores = scores.ToVector();
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
            throw new InvalidOperationException("HaralickSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("HaralickSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("HaralickSelector has not been fitted.");

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
