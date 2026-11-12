using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Multivariate;

/// <summary>
/// Conditional Mutual Information Maximization (CMIM) for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// CMIM uses a max-min criterion to select features. For each candidate, it computes
/// the minimum conditional mutual information with respect to all already-selected
/// features, then selects the candidate with the maximum of these minima.
/// </para>
/// <para><b>For Beginners:</b> CMIM is cautious about redundancy. When considering a
/// new feature, it looks at how much information it provides about the target given
/// each already-selected feature (conditioning). It takes the worst case (minimum)
/// and picks the feature whose worst case is best. This ensures every selected feature
/// adds value no matter which other feature you consider.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class CMIM<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _relevanceScores;
    private Vector<T>? _cmimScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? RelevanceScores => _relevanceScores;
    public Vector<T>? CmimScores => _cmimScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public CMIM(
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
            "CMIM requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var minimum = relevance;
        var (selected, scores) = DiscreteInformation<T>.Greedy(p, _nFeaturesToSelect, chosen =>
        {
            if (chosen.Count > 0)
                minimum = Engine.TensorMin(minimum, info.RelevanceGiven(chosen[chosen.Count - 1]));
            return minimum;
        });

        // a chosen feature keeps the score it won with; the others hold their running minimum
        var chosenMask = FeatureSelectionTensorOps.ScatterToFeatures(
            FilterStatisticTensorOps.Filled(new[] { selected.Length }, NumOps.One), selected, p, NumOps.Zero);
        _relevanceScores = relevance.ToVector();
        _cmimScores = Engine.TensorWhere(chosenMask, FilterStatisticTensorOps.AsTensor(scores), minimum).ToVector();
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
            throw new InvalidOperationException("CMIM has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("CMIM does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("CMIM has not been fitted.");

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
