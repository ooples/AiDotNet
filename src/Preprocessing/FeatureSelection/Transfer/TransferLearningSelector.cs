using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Transfer;

/// <summary>
/// Transfer Learning-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses feature importance learned from a source domain to guide feature
/// selection in a target domain. Particularly useful when the target domain
/// has limited data but the source domain has abundant labeled data.
/// </para>
/// <para><b>For Beginners:</b> Sometimes we have lots of data from one situation
/// (like medical records from one hospital) but little data from another
/// (a new hospital). Transfer learning uses what we learned about important
/// features from the first situation to help with the second.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class TransferLearningSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly Vector<T>? _sourceImportances;
    private readonly double _transferWeight;

    private Vector<T>? _targetImportances;
    private Vector<T>? _combinedImportances;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public double TransferWeight => _transferWeight;
    public Vector<T>? TargetImportances => _targetImportances;
    public Vector<T>? CombinedImportances => _combinedImportances;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public TransferLearningSelector(
        int nFeaturesToSelect = 10,
        Vector<T>? sourceImportances = null,
        double transferWeight = 0.5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (transferWeight < 0 || transferWeight > 1)
            throw new ArgumentException("Transfer weight must be between 0 and 1.", nameof(transferWeight));

        _nFeaturesToSelect = nFeaturesToSelect;
        _sourceImportances = sourceImportances;
        _transferWeight = transferWeight;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "TransferLearningSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // Compute target domain importances: |Pearson r| with the target (0 for a constant column or target)
        var targetImportances = Engine.TensorAbs(PreprocessingTensorOps.TargetCorrelations(
            Tensor<T>.FromMatrix(data), Tensor<T>.FromVector(target)));
        _targetImportances = targetImportances.ToVector();

        // Combine with source importances if available
        var combined = targetImportances;
        if (_sourceImportances is not null && _sourceImportances.Length == p && p > 0)
        {
            // Normalize both by their maximum (+1e-10)
            var source = Tensor<T>.FromVector(_sourceImportances);
            T eps = NumOps.FromDouble(1e-10);
            var normSource = Engine.TensorDivideScalar(source, NumOps.Add(Engine.TensorMaxValue(source), eps));
            var normTarget = Engine.TensorDivideScalar(targetImportances, NumOps.Add(Engine.TensorMaxValue(targetImportances), eps));
            combined = Engine.TensorAdd(
                Engine.TensorMultiplyScalar(normSource, NumOps.FromDouble(_transferWeight)),
                Engine.TensorMultiplyScalar(normTarget, NumOps.FromDouble(1 - _transferWeight)));
        }

        _combinedImportances = combined.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(combined, Math.Min(_nFeaturesToSelect, p));

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
            throw new InvalidOperationException("TransferLearningSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("TransferLearningSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("TransferLearningSelector has not been fitted.");

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
