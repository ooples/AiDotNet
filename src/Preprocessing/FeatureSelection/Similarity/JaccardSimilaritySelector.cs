using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Similarity;

/// <summary>
/// Jaccard Similarity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Jaccard similarity with the target after
/// binarization, measuring the intersection over union of positive values.
/// </para>
/// <para><b>For Beginners:</b> Jaccard similarity measures overlap between sets.
/// After converting features to binary (above/below threshold), it compares
/// which points are "high" in both feature and target simultaneously.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class JaccardSimilaritySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _jaccardScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? JaccardScores => _jaccardScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public JaccardSimilaritySelector(
        int nFeaturesToSelect = 10,
        int[]? columnIndices = null)
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
            "JaccardSimilaritySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        Tensor<T> scores;
        if (n == 0)
        {
            scores = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        }
        else
        {
            // Binarize the target and every column at their medians (value >= median)
            var x = Tensor<T>.FromMatrix(data);
            var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
            var yBinary = engine.TensorBroadcastTo(
                engine.TensorGreaterEqual(y, engine.TensorBroadcastTo(SelectorTensorOps.ColumnMedian(y, keepDims: true), new[] { n, 1 })),
                new[] { n, p });
            var xBinary = engine.TensorGreaterEqual(x, engine.TensorBroadcastTo(SelectorTensorOps.ColumnMedian(x, keepDims: true), new[] { n, p }));

            // Jaccard similarity |A and B| / |A or B|, 0 for an empty union
            var both = engine.TensorMultiply(xBinary, yBinary);
            var intersection = engine.ReduceSum(both, new[] { 0 }, false);
            var union = engine.ReduceSum(engine.TensorSubtract(engine.TensorAdd(xBinary, yBinary), both), new[] { 0 }, false);
            scores = SelectorTensorOps.ZeroUnless(engine.TensorGreaterThan(union, NumOps.Zero), engine.TensorDivide(intersection, union));
        }

        _jaccardScores = scores.ToVector();

        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("JaccardSimilaritySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("JaccardSimilaritySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JaccardSimilaritySelector has not been fitted.");

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
