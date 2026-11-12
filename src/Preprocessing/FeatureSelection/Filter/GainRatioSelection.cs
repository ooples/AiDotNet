using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gain Ratio Feature Selection (normalized Information Gain).
/// </summary>
/// <remarks>
/// <para>
/// Gain Ratio normalizes information gain by the intrinsic information of the
/// feature (split information). This corrects for the bias of information gain
/// towards features with many values.
/// </para>
/// <para><b>For Beginners:</b> Information gain tends to favor features with
/// many unique values (like IDs), even if they're not truly useful. Gain ratio
/// fixes this by dividing the information gain by how "spread out" the feature
/// values are. This gives fairer scores across features with different numbers
/// of unique values.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GainRatioSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _gainRatioScores;
    private Vector<T>? _informationGain;
    private Vector<T>? _splitInfo;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? GainRatioScores => _gainRatioScores;
    public Vector<T>? InformationGain => _informationGain;
    public Vector<T>? SplitInfo => _splitInfo;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GainRatioSelection(
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
            "GainRatioSelection requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize features: min(trunc((x - min) / range * (bins - 1)), bins - 1), 0 for a constant feature
        var featureBins = PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins);
        var joint = FeatureSelectionTensorOps.JointBinCounts(featureBins, classOneHot, _nBins);

        // IG = H(Y) - H(Y|X), split info H(X), gain ratio IG / split info (0 when split info is not above 1e-10)
        var splitInfo = FeatureSelectionTensorOps.BinEntropyNats(joint, n, out var conditionalEntropy);
        var informationGain = engine.TensorNegate(engine.TensorAddScalar(conditionalEntropy, NumOps.Negate(targetEntropy)));
        var usable = engine.TensorGreaterThan(splitInfo, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var gainRatioScores = engine.TensorWhere(usable, engine.TensorDivide(informationGain,
            engine.TensorWhere(usable, splitInfo, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))), new Tensor<T>(new[] { p }));
        _informationGain = informationGain.ToVector();
        _splitInfo = splitInfo.ToVector();
        _gainRatioScores = gainRatioScores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(gainRatioScores, numToSelect);

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
            throw new InvalidOperationException("GainRatioSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GainRatioSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GainRatioSelection has not been fitted.");

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
