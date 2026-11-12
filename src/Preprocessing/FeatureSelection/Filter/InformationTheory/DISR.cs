using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Double Input Symmetric Relevance (DISR) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// DISR extends mutual information by considering symmetric relevance between
/// pairs of features and the target. It uses the concept of joint symmetric
/// uncertainty to measure feature relevance.
/// </para>
/// <para><b>For Beginners:</b> DISR looks at how pairs of features together
/// relate to the target, using a normalized measure that accounts for both
/// directions of information flow. This helps find features that complement
/// each other in predicting the target.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class DISR<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _relevanceScores;
    private Vector<T>? _disrScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? RelevanceScores => _relevanceScores;
    public Vector<T>? DISRScores => _disrScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public DISR(
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
            "DISR requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var info = DiscreteInformation<T>.ForTarget(
            FeatureSelectionTensorOps.RangeBins(FilterStatisticTensorOps.AsTensor(data), _nBins), _nBins, target);
        var relevance = info.Relevance();
        var symmetricUncertainty = FilterStatisticTensorOps.GuardedDivide(
            Engine.TensorMultiplyScalar(relevance, NumOps.FromDouble(2)),
            Engine.TensorAddScalar(info.FeatureEntropy(), info.TargetEntropy), 0, NumOps.Zero);
        // mean symmetric relevance with the selected features, kept as a running mean (see
        // DiscreteInformation.RunningMean)
        var meanRelevance = FilterStatisticTensorOps.FilledLike(relevance, NumOps.Zero);
        var (selected, scores) = DiscreteInformation<T>.Greedy(p, _nFeaturesToSelect, chosen =>
        {
            if (chosen.Count == 0)
                return symmetricUncertainty;
            int last = chosen[chosen.Count - 1];
            meanRelevance = DiscreteInformation<T>.RunningMean(meanRelevance, FilterStatisticTensorOps.GuardedDivide(
                info.JointRelevance(last), info.PairTargetEntropy(last), 0, NumOps.Zero), chosen.Count);
            return meanRelevance;
        });

        _relevanceScores = symmetricUncertainty.ToVector();
        _disrScores = scores;
        _selectedIndices = selected;
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
            throw new InvalidOperationException("DISR has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("DISR does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("DISR has not been fitted.");

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
