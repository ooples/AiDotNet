using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Similarity;

/// <summary>
/// Spearman Rank Correlation based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Spearman rank correlation with the target,
/// measuring monotonic relationships (not necessarily linear).
/// </para>
/// <para><b>For Beginners:</b> Spearman correlation works on ranks rather than raw values.
/// It detects if one variable consistently increases when another increases, even if
/// the relationship isn't a straight line. It's more robust to outliers than Pearson.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpearmanCorrelationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;

    private Vector<T>? _correlations;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? Correlations => _correlations;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SpearmanCorrelationSelector(
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
            "SpearmanCorrelationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // |Spearman rho|: Pearson correlation of the 1-based average ranks, 0 where sqrt(sxx * syy) <= 1e-10
        var xRanks = SelectorTensorOps.AverageRanks(Tensor<T>.FromMatrix(data));
        var yRanks = Engine.Reshape(
            SelectorTensorOps.AverageRanks(Engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 })), new[] { n });
        var scores = Engine.TensorAbs(PreprocessingTensorOps.CorrelationsWithDenominatorGuard(xRanks, yRanks));
        _correlations = scores.ToVector();
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
            throw new InvalidOperationException("SpearmanCorrelationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpearmanCorrelationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SpearmanCorrelationSelector has not been fitted.");

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
