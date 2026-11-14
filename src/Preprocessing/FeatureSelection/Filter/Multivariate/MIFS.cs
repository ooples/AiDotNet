using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Mutual Information Feature Selection (MIFS) with redundancy penalty.
/// </summary>
/// <remarks>
/// <para>
/// MIFS iteratively selects features that have high mutual information with the target
/// while penalizing redundancy with already-selected features. The beta parameter
/// controls the strength of the redundancy penalty.
/// </para>
/// <para><b>For Beginners:</b> MIFS greedily picks features that tell you a lot about
/// the target. But it also subtracts a penalty if the new feature overlaps with what
/// you already know from selected features. The beta parameter controls how harsh
/// this penalty is: 0 = no penalty (just pick high MI), 1 = full penalty.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MIFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _beta;

    private Vector<T>? _relevanceScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? RelevanceScores => _relevanceScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MIFS(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double beta = 1.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _beta = beta;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "MIFS requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int p = data.Columns;
        var info = DiscreteInformation<T>.ForTarget(
            FeatureScoringTensorOps.ScaledBins(FilterStatisticTensorOps.AsTensor(data), _nBins), _nBins, target);
        var relevance = info.Relevance();
        T beta = NumOps.FromDouble(_beta);
        var redundancy = FilterStatisticTensorOps.FilledLike(relevance, NumOps.Zero);
        var (selected, _) = DiscreteInformation<T>.Greedy(p, _nFeaturesToSelect, chosen =>
        {
            if (chosen.Count == 0)
                return relevance;
            redundancy = Engine.TensorAdd(redundancy, info.FeatureMutualInformation(chosen[chosen.Count - 1]));
            return Engine.TensorSubtract(relevance, Engine.TensorMultiplyScalar(redundancy, beta));
        });

        _relevanceScores = relevance.ToVector();
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
            throw new InvalidOperationException("MIFS has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MIFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MIFS has not been fitted.");

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
