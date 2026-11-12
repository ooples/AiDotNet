using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Similarity;

/// <summary>
/// Cosine Similarity based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their cosine similarity with the target,
/// measuring directional alignment rather than magnitude.
/// </para>
/// <para><b>For Beginners:</b> Cosine similarity measures the angle between two
/// vectors, ignoring their length. Features pointing in similar directions
/// as the target are selected, which is useful when scale doesn't matter.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CosineSimilaritySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _cosineSimilarities;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? CosineSimilarities => _cosineSimilarities;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CosineSimilaritySelector(
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
            "CosineSimilaritySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var x = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        T epsilon = NumOps.FromDouble(1e-10);

        // |x . y| / (||x|| ||y||), 0 where either norm is not above 1e-10
        T yNorm = NumOps.Sqrt(engine.TensorSum(engine.TensorMultiply(y, y)));
        var xNorm = engine.TensorSqrt(SelectorTensorOps.ColumnSumOfSquares(x));
        var dot = engine.ReduceSum(engine.TensorMultiply(x, engine.TensorBroadcastTo(y, new[] { n, p })), new[] { 0 }, false);
        var cosine = engine.TensorAbs(engine.TensorDivide(dot, engine.TensorMultiplyScalar(xNorm, yNorm)));
        var valid = NumOps.GreaterThan(yNorm, epsilon)
            ? engine.TensorGreaterThan(xNorm, epsilon)
            : PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        var scores = SelectorTensorOps.ZeroUnless(valid, cosine);
        _cosineSimilarities = scores.ToVector();

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
            throw new InvalidOperationException("CosineSimilaritySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CosineSimilaritySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CosineSimilaritySelector has not been fitted.");

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
