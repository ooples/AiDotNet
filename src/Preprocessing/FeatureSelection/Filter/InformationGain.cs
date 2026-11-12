using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Information Gain (Mutual Information) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Information Gain measures the reduction in entropy of the target variable when
/// a feature is known. It's equivalent to mutual information and quantifies how much
/// information about the target a feature provides.
/// </para>
/// <para><b>For Beginners:</b> Information Gain asks: "How much does knowing this
/// feature reduce my uncertainty about the target?" If knowing the feature value
/// tells you a lot about what the target will be, it has high information gain.
/// It's measured in bits (or nats, depending on the logarithm base).
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InformationGain<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _informationGains;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? InformationGains => _informationGains;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public InformationGain(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InformationGain requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Target entropy (nats) over the rounded class labels
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, classes.Length);
        T targetEntropy = FeatureSelectionTensorOps.ClassEntropyNats(classOneHot);

        // Discretize features: min(trunc((x - min) / range * bins), bins - 1), one bin for a constant feature
        var featureBins = FeatureSelectionTensorOps.RangeBinOneHot(Tensor<T>.FromMatrix(data), _nBins, _nBins);
        FeatureSelectionTensorOps.BinEntropyNats(
            FeatureSelectionTensorOps.JointBinCounts(featureBins, classOneHot, _nBins), n, out var conditionalEntropy);

        // Information Gain = H(Y) - H(Y|X)
        var informationGains = engine.TensorNegate(engine.TensorAddScalar(conditionalEntropy, NumOps.Negate(targetEntropy)));
        _informationGains = informationGains.ToVector();

        // Select top features by Information Gain
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(informationGains, numToSelect);

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
            throw new InvalidOperationException("InformationGain has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InformationGain does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InformationGain has not been fitted.");

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
