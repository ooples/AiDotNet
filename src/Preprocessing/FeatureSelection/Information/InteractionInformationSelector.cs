using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Information;

/// <summary>
/// Interaction Information-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their interaction information with the target,
/// capturing synergistic and redundant relationships between feature pairs.
/// </para>
/// <para><b>For Beginners:</b> Interaction information measures whether two
/// features together tell us more (synergy) or less (redundancy) about the
/// target than we'd expect from their individual contributions. This helps
/// find features that work well together.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class InteractionInformationSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _interactionScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? InteractionScores => _interactionScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public InteractionInformationSelector(
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
            "InteractionInformationSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        if (n == 0)
            throw new InvalidOperationException("Sequence contains no elements");

        var engine = Engine;

        // Discretize all features and target
        var xBins = FeatureSelectionInformationOps.EqualWidthBins(Tensor<T>.FromMatrix(data), _nBins);
        var yBins = FeatureSelectionInformationOps.EqualWidthBins(target, _nBins);

        // Compute individual mutual information I(X_j;Y) in bits
        var mi = FeatureSelectionInformationOps.ToBits(FeatureSelectionInformationOps.MutualInformation(
            FeatureSelectionInformationOps.JointCounts<T>(xBins, _nBins, yBins, _nBins), n));

        // Joint mutual information I(X_j,X_k;Y) for every pair, one row of partners per feature
        var jointRows = new Tensor<T>[p];
        for (int j = 0; j < p; j++)
        {
            var pairs = FeatureSelectionInformationOps.Combine(engine.TensorSlice(xBins, new[] { 0, j }, new[] { n, 1 }), xBins, _nBins);
            jointRows[j] = FeatureSelectionInformationOps.ToBits(FeatureSelectionInformationOps.MutualInformation(
                FeatureSelectionInformationOps.JointCounts<T>(pairs, _nBins * _nBins, yBins, _nBins), n));
        }

        // I(X;Y;Z) = I(X;Y) + I(Z;Y) - I(X,Z;Y); each feature adds its average |interaction| with the others
        var scores = mi;
        if (p > 1)
        {
            var joint = engine.TensorStack(jointRows, 0);
            var interaction = engine.TensorAbs(engine.TensorSubtract(
                engine.TensorAdd(
                    engine.TensorBroadcastTo(engine.Reshape(mi, new[] { p, 1 }), new[] { p, p }),
                    engine.TensorBroadcastTo(engine.Reshape(mi, new[] { 1, p }), new[] { p, p })),
                joint));
            var offDiagonal = PreprocessingTensorOps.OneMinus(engine.TensorEye<T>(p));
            var total = engine.ReduceSum(engine.TensorMultiply(interaction, offDiagonal), new[] { 1 }, false);
            scores = engine.TensorAdd(mi, engine.TensorDivideScalar(total, FeatureSelectionTensorOps.Count<T>(p - 1)));
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
            throw new InvalidOperationException("InteractionInformationSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("InteractionInformationSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("InteractionInformationSelector has not been fitted.");

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
