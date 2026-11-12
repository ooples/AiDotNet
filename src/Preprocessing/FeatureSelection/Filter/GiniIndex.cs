using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Gini Index feature selection for measuring impurity reduction.
/// </summary>
/// <remarks>
/// <para>
/// The Gini Index measures the impurity of a split based on the probability of misclassifying
/// a randomly chosen element. Features that result in lower weighted Gini impurity after
/// splitting are considered more informative.
/// </para>
/// <para><b>For Beginners:</b> Imagine randomly guessing the class of an item. The Gini Index
/// measures how often you'd guess wrong. A feature that separates classes well will have
/// a low Gini Index (few wrong guesses). This is the same metric used by decision trees
/// like CART to choose splits.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class GiniIndex<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _giniReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? GiniReductions => _giniReductions;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public GiniIndex(
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
            "GiniIndex requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Overall Gini impurity over the rounded class labels
        PreprocessingTensorOps.ClassLabels(target, out var classes, out var classIndex);
        var classOneHot = PreprocessingTensorOps.ClassOneHot<T>(classIndex, classes.Length);
        T overallGini = FeatureSelectionTensorOps.GiniImpurity(classOneHot);

        // Bin features: min(trunc((x - min) / range * bins), bins - 1), one bin for a constant feature
        var featureBins = FeatureSelectionTensorOps.RangeBinOneHot(Tensor<T>.FromMatrix(data), _nBins, _nBins);
        var weightedGini = FeatureSelectionTensorOps.WeightedBinGini(
            FeatureSelectionTensorOps.JointBinCounts(featureBins, classOneHot, _nBins), n);

        // Gini reduction max(0, G - weighted G) (higher is better)
        var giniReductions = engine.TensorMax(
            engine.TensorNegate(engine.TensorAddScalar(weightedGini, NumOps.Negate(overallGini))), NumOps.Zero);
        _giniReductions = giniReductions.ToVector();

        // Select top features by Gini reduction
        int nToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(giniReductions, nToSelect);

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
            throw new InvalidOperationException("GiniIndex has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("GiniIndex does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("GiniIndex has not been fitted.");

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
