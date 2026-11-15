using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Robust;

/// <summary>
/// Trimmed Mean-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses trimmed statistics (removing extreme values) to compute robust feature
/// importance scores that are less affected by outliers.
/// </para>
/// <para><b>For Beginners:</b> When computing feature statistics, extreme values
/// (outliers) can skew results. Trimmed mean cuts off the highest and lowest
/// values before computing, giving you a more reliable picture of typical feature
/// behavior without outlier influence.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TrimmedMeanSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly double _trimProportion;

    private Vector<T>? _trimmedVariances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double TrimProportion => _trimProportion;
    public Vector<T>? TrimmedVariances => _trimmedVariances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TrimmedMeanSelector(
        int nFeaturesToSelect = 10,
        double trimProportion = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (trimProportion < 0 || trimProportion >= 0.5)
            throw new ArgumentException("Trim proportion must be between 0 and 0.5.", nameof(trimProportion));

        _nFeaturesToSelect = nFeaturesToSelect;
        _trimProportion = trimProportion;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        int trimCount = (int)(n * _trimProportion);
        int trimmedN = n - 2 * trimCount;

        var engine = Engine;
        var variances = PreprocessingTensorOps.Filled(new[] { p }, NumOps.Zero);
        if (trimmedN > 1)
        {
            // Sample variance (n - 1) of the values left after trimming trimCount from each end of every column.
            var sorted = engine.TensorSort(Tensor<T>.FromMatrix(data), 0, false).Values;
            var kept = engine.TensorSlice(sorted, new[] { trimCount, 0 }, new[] { trimmedN, p });
            var centered = engine.TensorSubtract(kept, engine.ReduceMean(kept, new[] { 0 }, true));
            variances = engine.TensorDivideScalar(
                engine.ReduceSum(engine.TensorMultiply(centered, centered), new[] { 0 }, false),
                NumOps.FromDouble(trimmedN - 1));
        }

        _trimmedVariances = variances.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(variances, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrimmedMeanSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TrimmedMeanSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TrimmedMeanSelector has not been fitted.");

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
