using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Gain;

/// <summary>
/// Information Gain Ratio based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on information gain ratio, which normalizes
/// information gain by the feature's intrinsic information to avoid
/// bias toward features with many distinct values.
/// </para>
/// <para><b>For Beginners:</b> Information gain measures how much knowing a
/// feature reduces uncertainty about the target. But features with many values
/// (like IDs) can have artificially high gain. Gain ratio fixes this by dividing
/// by the feature's own entropy, giving a fair comparison across features.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InformationGainRatioSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _gainRatios;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? GainRatios => _gainRatios;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public InformationGainRatioSelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InformationGainRatioSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // the target is binned on the same grid as the features; the ratio is the same in bits or nats, and the
        // intrinsic-information floor of 1e-10 bits is 1e-10 ln 2 nats
        var targetCodes = FeatureScoringTensorOps.RangeBins(
            Engine.Reshape(FilterStatisticTensorOps.AsTensor(target), new[] { n, 1 }), _nBins);
        var info = new DiscreteInformation<T>(
            FeatureScoringTensorOps.RangeBins(FilterStatisticTensorOps.AsTensor(data), _nBins), _nBins,
            Engine.Reshape(FeatureScoringTensorOps.OneHot<T>(targetCodes, _nBins), new[] { n, _nBins }));
        var ratios = FilterStatisticTensorOps.GuardedDivide(
            info.Relevance(), info.FeatureEntropy(), 1e-10 * Math.Log(2), NumOps.Zero);
        _gainRatios = ratios.ToVector();
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(ratios, Math.Min(_nFeaturesToSelect, p));
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
            throw new InvalidOperationException("InformationGainRatioSelector has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InformationGainRatioSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InformationGainRatioSelector has not been fitted.");

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
