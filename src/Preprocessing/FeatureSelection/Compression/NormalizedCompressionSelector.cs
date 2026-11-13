using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Compression;

/// <summary>
/// Normalized Compression Distance based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their normalized compression distance to the target,
/// approximating Kolmogorov complexity for feature relevance.
/// </para>
/// <para><b>For Beginners:</b> This measures how much "shared information" exists
/// between a feature and the target. Features that compress well together with
/// the target share more mutual information.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class NormalizedCompressionSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _ncdScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? NCDScores => _ncdScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public NormalizedCompressionSelector(
        int nFeaturesToSelect = 10,
        int nBins = 20,
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
            "NormalizedCompressionSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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
        var X = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });

        // Equal-width histograms of every feature and of the target ((near-)constant ones have entropy 0)
        var oneHotX = PreprocessingTensorOps.EqualWidthBinOneHot(X, _nBins, out var xVarying);
        var oneHotY = PreprocessingTensorOps.EqualWidthBinOneHot(y, _nBins, out var yVaryingMask);
        bool yVarying = PreprocessingTensorOps.CountNonZero(yVaryingMask) > 0;
        var entropyX = engine.TensorMultiply(
            FeatureSelectionTensorOps.EntropyBits(FeatureSelectionTensorOps.BinCounts(oneHotX, _nBins), n), xVarying);
        var entropyY = yVarying
            ? FeatureSelectionTensorOps.EntropyBits(FeatureSelectionTensorOps.BinCounts(oneHotY, _nBins), n)
            : new Tensor<T>(new[] { 1 });
        var entropyYRow = engine.TensorBroadcastTo(entropyY, new[] { p });

        // Joint entropy H(X, Y) from the joint histogram; with a constant X it is H(Y), with a constant Y it is H(X)
        Tensor<T> jointEntropy;
        if (yVarying)
        {
            // Counted exactly (a Half matmul of the 0/1 one-hots stops counting at 2048)
            var jointCounts = engine.Reshape(
                FeatureSelectionTensorOps.JointBinCounts(oneHotX, oneHotY, _nBins), new[] { p, _nBins * _nBins });
            jointEntropy = engine.TensorWhere(xVarying, FeatureSelectionTensorOps.EntropyBits(jointCounts, n), entropyYRow);
        }
        else
        {
            jointEntropy = entropyX;
        }

        // NCD approximation: lower NCD means more similarity
        // We use mutual information instead: higher means more relevant
        var mi = engine.TensorMax(
            engine.TensorSubtract(engine.TensorAdd(entropyX, entropyYRow), jointEntropy), NumOps.Zero);
        _ncdScores = mi.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(mi, numToSelect);

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
            throw new InvalidOperationException("NormalizedCompressionSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("NormalizedCompressionSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("NormalizedCompressionSelector has not been fitted.");

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
