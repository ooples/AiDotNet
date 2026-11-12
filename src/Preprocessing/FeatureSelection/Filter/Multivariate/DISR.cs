using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Double Input Symmetrical Relevance (DISR) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// DISR uses a symmetrical relevance measure that considers both feature-target and
/// feature-feature relationships. It maximizes a normalized joint relevance score
/// that accounts for redundancy in a principled way.
/// </para>
/// <para><b>For Beginners:</b> DISR balances relevance and redundancy by looking at
/// how much two features together tell you about the target, normalized by their
/// total information content. It's like asking: "Of all the information in this
/// feature pair, what fraction is actually useful for prediction?"
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
    public Vector<T>? RelevanceScores => _relevanceScores;
    public Vector<T>? DisrScores => _disrScores;
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
            FeatureSelectionTensorOps.ScaledBins(FilterStatisticTensorOps.AsTensor(data), _nBins), _nBins, target);
        var relevance = info.Relevance();
        var symmetricUncertainty = FilterStatisticTensorOps.GuardedDivide(
            Engine.TensorMultiplyScalar(relevance, NumOps.FromDouble(2)),
            Engine.TensorAddScalar(info.FeatureEntropy(), info.TargetEntropy), 1e-10, NumOps.Zero);
        T one = NumOps.One;
        var sum = FilterStatisticTensorOps.FilledLike(relevance, NumOps.Zero);
        var steps = sum;
        var (selected, scores) = DiscreteInformation<T>.Greedy(p, _nFeaturesToSelect, chosen =>
        {
            if (chosen.Count == 0)
                return symmetricUncertainty;
            int last = chosen[chosen.Count - 1];
            sum = Engine.TensorAdd(sum, FilterStatisticTensorOps.GuardedDivide(
                info.JointRelevance(last), info.PairTargetEntropy(last), 1e-10, NumOps.Zero));
            steps = Engine.TensorAddScalar(steps, one);
            return Engine.TensorDivide(sum, steps);
        });

        _relevanceScores = relevance.ToVector();
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
