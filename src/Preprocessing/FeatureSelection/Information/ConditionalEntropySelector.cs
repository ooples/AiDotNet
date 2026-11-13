using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Information;

/// <summary>
/// Conditional Entropy-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that minimize the conditional entropy of the target
/// given the feature, maximizing the information about the target.
/// </para>
/// <para><b>For Beginners:</b> Conditional entropy measures the uncertainty
/// remaining about the target after knowing a feature's value. Features that
/// reduce uncertainty the most (lower conditional entropy) are more useful.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ConditionalEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _conditionalEntropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? ConditionalEntropies => _conditionalEntropies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public ConditionalEntropySelector(
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
            "ConditionalEntropySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize every feature and the target into equal-width bins
        var xBins = FeatureSelectionInformationOps.EqualWidthBins(Tensor<T>.FromMatrix(data), _nBins);
        var yBins = FeatureSelectionInformationOps.EqualWidthBins(target, _nBins);

        // H(Y|X) = sum over x of P(x) * H(Y|X=x), in bits
        var counts = FeatureSelectionInformationOps.JointCounts<T>(xBins, _nBins, yBins, _nBins);
        var entropies = FeatureSelectionInformationOps.ToBits(FeatureSelectionInformationOps.ConditionalEntropy(counts, n));
        _conditionalEntropies = entropies.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        // Select features with LOWEST conditional entropy (most information)
        _selectedIndices = PreprocessingTensorOps.BottomKIndicesSorted(entropies, numToSelect);

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
            throw new InvalidOperationException("ConditionalEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("ConditionalEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("ConditionalEntropySelector has not been fitted.");

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
