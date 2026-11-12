using AiDotNet.Preprocessing.FeatureSelection.Helpers;
using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Normalized Mutual Information for scale-invariant dependency measurement.
/// </summary>
/// <remarks>
/// <para>
/// Normalized Mutual Information (NMI) divides mutual information by entropy
/// to get a value between 0 and 1, making comparisons across features more fair.
/// </para>
/// <para><b>For Beginners:</b> Regular mutual information can be higher for
/// features with more distinct values. NMI normalizes this so you can fairly
/// compare features with different numbers of unique values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NormalizedMutualInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly NormalizationMethod _normalization;

    public enum NormalizationMethod { Arithmetic, Geometric, Min, Max }

    private Vector<T>? _nmiScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? NMIScores => _nmiScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public NormalizedMutualInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        NormalizationMethod normalization = NormalizationMethod.Arithmetic,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _normalization = normalization;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "NormalizedMutualInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;

        // the target's distinct values are its classes
        var info = new DiscreteInformation<T>(
            FeatureSelectionTensorOps.RangeBins(FilterStatisticTensorOps.AsTensor(data), _nBins), _nBins,
            StatisticalTestHelper.ClassOneHot(target, out _));
        var hX = info.FeatureEntropy();
        T hY = info.TargetEntropy;
        var norm = _normalization switch
        {
            NormalizationMethod.Geometric => Engine.TensorSqrt(Engine.TensorMultiplyScalar(hX, hY)),
            NormalizationMethod.Min => Engine.TensorMin(hX, hY),
            NormalizationMethod.Max => Engine.TensorMax(hX, hY),
            _ => Engine.TensorMultiplyScalar(Engine.TensorAddScalar(hX, hY), NumOps.FromDouble(0.5))
        };
        var scores = FilterStatisticTensorOps.GuardedDivide(info.Relevance(), norm, 1e-10, NumOps.Zero);
        _nmiScores = scores.ToVector();
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("NormalizedMutualInformation has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NormalizedMutualInformation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NormalizedMutualInformation has not been fitted.");

        var mask = new bool[_nInputFeatures];
        foreach (int idx in _selectedIndices)
            mask[idx] = true;

        return mask;
    }

    /// <inheritdoc />
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        if (_selectedIndices is null) return Array.Empty<string>();

        if (inputFeatureNames is null)
            return _selectedIndices.Select(i => $"Feature{i}").ToArray();

        return _selectedIndices
            .Where(i => i < inputFeatureNames.Length)
            .Select(i => inputFeatureNames[i])
            .ToArray();
    }
}
