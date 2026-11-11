using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Classification;

/// <summary>
/// Linear Discriminant Analysis Projection based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their contribution to class separation using
/// the ratio of between-class to within-class variance.
/// </para>
/// <para><b>For Beginners:</b> LDA finds directions that best separate classes.
/// This selector measures how much each feature contributes to these separating
/// directions, keeping features that help distinguish between groups.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LDAProjectionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _ldaScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? LDAScores => _ldaScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public LDAProjectionSelector(
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
            "LDAProjectionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = Tensor<T>.FromMatrix(data);
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        int k = classes.Length;
        var membership = PreprocessingTensorOps.ClassOneHot<T>(classIndex, k);                 // [n, k]
        var classCounts = engine.ReduceSum(membership, new[] { 0 }, false);                     // [k]

        // Global mean and class means
        var globalMean = engine.ReduceMean(X, new[] { 0 }, true);                               // [1, p]
        var classMeans = engine.TensorDivide(
            engine.TensorMatMul(engine.TensorTranspose(membership), X), engine.Reshape(classCounts, new[] { k, 1 })); // [k, p]

        // LDA criterion per feature (Sb / Sw ratio approximation)
        var withinClassVar = engine.ReduceSum(
            engine.TensorSquare(engine.TensorSubtract(X, engine.TensorMatMul(membership, classMeans))), new[] { 0 }, false);
        var betweenClassVar = engine.Reshape(
            engine.TensorMatMul(
                engine.Reshape(classCounts, new[] { 1, k }),
                engine.TensorSquare(engine.TensorSubtract(classMeans, globalMean))),
            new[] { p });

        var scores = engine.TensorWhere(
            engine.TensorGreaterThan(withinClassVar, NumOps.FromDouble(1e-10)),
            engine.TensorDivide(betweenClassVar, withinClassVar),
            new Tensor<T>(new[] { p }));
        _ldaScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("LDAProjectionSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LDAProjectionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("LDAProjectionSelector has not been fitted.");

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
