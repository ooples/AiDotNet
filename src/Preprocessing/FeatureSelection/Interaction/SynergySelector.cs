using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Interaction;

/// <summary>
/// Synergy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their synergistic information with other features,
/// measuring how features provide complementary information about the target.
/// </para>
/// <para><b>For Beginners:</b> Synergy occurs when two features together provide
/// more information than the sum of their individual information. This selector
/// finds features that work synergistically with others.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SynergySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _synergyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? SynergyScores => _synergyScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public SynergySelector(
        int nFeaturesToSelect = 10,
        int nBins = 5,
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
            "SynergySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var discretizedX = FeatureSelectionInformationOps.EqualWidthBinsExactRange(Tensor<T>.FromMatrix(data), _nBins);
        var discretizedY = FeatureSelectionInformationOps.EqualWidthBinsExactRange(
            engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins);

        // Compute individual mutual information I(X_j;Y) in bits
        var individualMI = engine.TensorMax(FeatureSelectionInformationOps.ToBits(FeatureSelectionInformationOps.MutualInformation(
            FeatureSelectionInformationOps.JointCounts<T>(discretizedX, _nBins, discretizedY, _nBins), n)), NumOps.Zero);

        // For each feature, compute maximum synergy with one of the first (at most 20) features
        int partners = Math.Min(p, 20); // Limit comparisons for efficiency
        var scores = individualMI;
        if (partners > 0)
        {
            var partnerBins = engine.TensorSlice(discretizedX, new[] { 0, 0 }, new[] { n, partners });
            var partnerMI = engine.Reshape(engine.TensorSlice(individualMI, new[] { 0 }, new[] { partners }), new[] { 1, partners });
            var synergyRows = new Tensor<T>[p];
            for (int j1 = 0; j1 < p; j1++)
            {
                // Joint MI I(X_j1, X_j2; Y), joint bin = bin_j1 * nBins + bin_j2
                var pairs = FeatureSelectionInformationOps.Combine(
                    engine.TensorSlice(discretizedX, new[] { 0, j1 }, new[] { n, 1 }), partnerBins, _nBins);
                var jointMI = engine.TensorMax(FeatureSelectionInformationOps.ToBits(FeatureSelectionInformationOps.MutualInformation(
                    FeatureSelectionInformationOps.JointCounts<T>(pairs, _nBins * _nBins, discretizedY, _nBins), n)), NumOps.Zero);

                // Synergy = I(X_j1, X_j2; Y) - max(I(X_j1;Y), I(X_j2;Y))
                var own = engine.TensorBroadcastTo(
                    engine.Reshape(engine.TensorSlice(individualMI, new[] { j1 }, new[] { 1 }), new[] { 1, 1 }), new[] { 1, partners });
                synergyRows[j1] = engine.TensorSubtract(engine.Reshape(jointMI, new[] { 1, partners }), engine.TensorMax(partnerMI, own));
            }

            // max(0, best synergy) over the partners other than the feature itself
            var synergy = engine.TensorMax(engine.TensorConcatenate(synergyRows, 0), NumOps.Zero);
            var notSelf = FeatureSelectionTensorOps.OneMinus(engine.TensorSlice(engine.TensorEye<T>(p), new[] { 0, 0 }, new[] { p, partners }));
            var maxSynergy = engine.ReduceMax(engine.TensorMultiply(synergy, notSelf), new[] { 1 }, false);
            scores = engine.TensorAdd(maxSynergy, individualMI); // Combine synergy with individual MI
        }

        _synergyScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("SynergySelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SynergySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("SynergySelector has not been fitted.");

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
