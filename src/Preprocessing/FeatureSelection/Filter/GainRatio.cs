using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gain Ratio for feature selection, normalizing Information Gain by feature entropy.
/// </summary>
/// <remarks>
/// <para>
/// Gain Ratio addresses Information Gain's bias toward features with many values
/// by dividing by the feature's intrinsic information (entropy of the feature itself).
/// </para>
/// <para><b>For Beginners:</b> Information Gain tends to favor features with many
/// unique values (like IDs). Gain Ratio fixes this by accounting for how complex
/// the feature is. A feature that tells you a lot AND is simple gets a higher score
/// than one that's complex but only slightly informative.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GainRatio<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _gainRatios;
    private Vector<T>? _informationGains;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? GainRatios => _gainRatios;
    public Vector<T>? InformationGains => _informationGains;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GainRatio(
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
            "GainRatio requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Class distribution (rounded target values) and target entropy (nats)
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        // Counts, entropies and ratios are in double and only the scores cast to T: Half counts past 65504 are
        // infinite (two separated classes of 70,000 rows gave a gain ratio of 0 instead of 1)
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, classes.Length).Cast<double>();
        double targetEntropy = FeatureScoringTensorOps.ClassEntropyNats(classOneHot);

        // Bin every feature: min(trunc((x - min) / range * bins), bins - 1), one bin for a constant feature
        var featureBins = FeatureScoringTensorOps.RangeBinOneHot(Tensor<T>.FromMatrix(data), _nBins, _nBins).Cast<double>();
        var joint = FeatureScoringTensorOps.JointBinCounts(featureBins, classOneHot, _nBins);

        // Information gain max(0, H(Y) - H(Y|X)) and gain ratio IG / H(X) (0 when H(X) is not above 1e-10)
        var intrinsicInfo = FeatureScoringTensorOps.BinEntropyNats(joint, n, out var conditionalEntropy);
        var informationGains = engine.TensorMax(
            engine.TensorNegate(engine.TensorAddScalar(conditionalEntropy, -targetEntropy)), 0.0);
        var usable = engine.TensorGreaterThan(intrinsicInfo, 1e-10);
        var gainRatios = engine.TensorWhere(usable, engine.TensorDivide(informationGains,
            engine.TensorWhere(usable, intrinsicInfo, PreprocessingTensorOps.Filled(new[] { p }, 1.0))),
            new Tensor<double>(new[] { p })).Cast<T>();
        _informationGains = informationGains.Cast<T>().ToVector();
        _gainRatios = gainRatios.ToVector();

        // Select top features
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(gainRatios, nToSelect);

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
            throw new InvalidOperationException("GainRatio has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GainRatio does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GainRatio has not been fitted.");

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
