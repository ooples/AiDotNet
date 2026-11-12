using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.InformationTheory;

/// <summary>
/// Interaction Information for feature selection.
/// </summary>
/// <remarks>
/// <para>
/// Interaction Information measures the amount of information bound up in a set of
/// variables that is not present in any subset. It can be positive (synergy) or
/// negative (redundancy), revealing higher-order dependencies.
/// </para>
/// <para><b>For Beginners:</b> This method finds features that work together in
/// unexpected ways. Sometimes two features alone are useless, but together they're
/// very predictive (synergy). Other times, features duplicate each other's information
/// (redundancy). Interaction Information captures both.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InteractionInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly bool _preferSynergy;

    private Vector<T>? _miScores;
    private Vector<T>? _interactionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public bool PreferSynergy => _preferSynergy;
    public Vector<T>? MIScores => _miScores;
    public Vector<T>? InteractionScores => _interactionScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public InteractionInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        bool preferSynergy = true,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _preferSynergy = preferSynergy;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InteractionInformation requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        T one = NumOps.One;
        var sum = FilterStatisticTensorOps.FilledLike(relevance, NumOps.Zero);
        var steps = sum;
        var (selected, scores) = DiscreteInformation<T>.Greedy(p, _nFeaturesToSelect, chosen =>
        {
            if (chosen.Count == 0)
                return relevance;
            int last = chosen[chosen.Count - 1];

            // I(Xj; Xs; Y) = I(Xj; Y) + I(Xs; Y) - I(Xj, Xs; Y)
            var interaction = Engine.TensorSubtract(
                Engine.TensorAdd(relevance, Engine.TensorBroadcastTo(
                    FeatureSelectionTensorOps.Entries(relevance, new[] { last }), new[] { p })),
                info.JointRelevance(last));
            sum = Engine.TensorAdd(sum, interaction);
            steps = Engine.TensorAddScalar(steps, one);
            var meanInteraction = Engine.TensorDivide(_preferSynergy ? sum : Engine.TensorAbs(sum), steps);
            return Engine.TensorSubtract(relevance, meanInteraction);
        });

        _miScores = relevance.ToVector();
        _interactionScores = scores;
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
            throw new InvalidOperationException("InteractionInformation has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InteractionInformation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionInformation has not been fitted.");

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
