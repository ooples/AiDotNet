using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Interaction;

/// <summary>
/// Interaction Strength based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their interaction strength with other features
/// in predicting the target.
/// </para>
/// <para><b>For Beginners:</b> Some features are more useful when combined with
/// others (like age and income together predicting spending). This selector
/// finds features that interact strongly with others.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InteractionStrengthSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _topInteractions;

    private Vector<T>? _interactionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int TopInteractions => _topInteractions;
    public Vector<T>? InteractionScores => _interactionScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public InteractionStrengthSelector(
        int nFeaturesToSelect = 10,
        int topInteractions = 5,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _topInteractions = topInteractions;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "InteractionStrengthSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (n == 0 && p > 1)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Standardize features (population std; a constant column is only centred)
        var centered = FilterStatisticTensorOps.CenterColumns(Tensor<T>.FromMatrix(data));
        var std = FeatureSelectionTensorOps.CenteredColumnStd(centered);
        var safeStd = engine.TensorWhere(engine.TensorLessThan(std, NumOps.FromDouble(1e-10)), PreprocessingTensorOps.Filled(new[] { p }, NumOps.One), std);
        var X = engine.TensorDivide(centered, engine.Reshape(safeStd, new[] { 1, p }));

        // Individual |correlations| with the target, as a [1, p] row
        var individual = engine.Reshape(engine.TensorAbs(FeatureSelectionTensorOps.ColumnCorrelations(X, y)), new[] { 1, p });

        // For each feature j1, how much each product term X_j1 * X_j2 correlates with the target beyond
        // the stronger of the two individual correlations
        var boostRows = new Tensor<T>[p];
        for (int j1 = 0; j1 < p; j1++)
        {
            var column = engine.TensorSlice(X, new[] { 0, j1 }, new[] { n, 1 });
            var products = engine.TensorMultiply(X, engine.TensorBroadcastTo(column, new[] { n, p }));
            var interactionCorr = engine.Reshape(engine.TensorAbs(FeatureSelectionTensorOps.ColumnCorrelations(products, y)), new[] { 1, p });
            var own = engine.TensorBroadcastTo(engine.TensorSlice(individual, new[] { 0, j1 }, new[] { 1, 1 }), new[] { 1, p });
            boostRows[j1] = engine.TensorSubtract(interactionCorr, engine.TensorMax(individual, own));
        }

        // Feature score = sum of its top positive interaction boosts (the feature itself excluded)
        var scores = new Tensor<T>(new[] { p });
        int top = Math.Min(_topInteractions, p);
        if (p > 0 && top > 0)
        {
            var boosts = engine.TensorMax(engine.TensorConcatenate(boostRows, 0), NumOps.Zero);
            boosts = engine.TensorMultiply(boosts, PreprocessingTensorOps.OneMinus(engine.TensorEye<T>(p)));
            var largest = engine.TensorTopK(boosts, top, 1, out _);
            scores = engine.ReduceSum(largest, new[] { 1 }, false);
        }

        _interactionScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("InteractionStrengthSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InteractionStrengthSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionStrengthSelector has not been fitted.");

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
