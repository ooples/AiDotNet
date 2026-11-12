using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Renyi Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their Renyi entropy, which generalizes Shannon entropy
/// with a tunable parameter alpha.
/// </para>
/// <para><b>For Beginners:</b> Renyi entropy is a family of entropy measures. With
/// alpha=1, it equals Shannon entropy. Different alpha values emphasize different
/// aspects of the distribution - low alpha focuses on rare events, high alpha on
/// common events.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class RenyiEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _alpha;

    private Vector<T>? _renyiEntropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public double Alpha => _alpha;
    public Vector<T>? RenyiEntropies => _renyiEntropies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public RenyiEntropySelector(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double alpha = 2.0,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (alpha <= 0 || Math.Abs(alpha - 1) < 1e-10)
            throw new ArgumentException("Alpha must be positive and not equal to 1.", nameof(alpha));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _alpha = alpha;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var floor = PreprocessingTensorOps.Tolerance<T>(1e-10);

        // Discretize: bin = min(bins - 1, trunc((v - min) / range * bins)), 0 when the column is constant
        var counts = FeatureSelectionTensorOps.BinCounts(
            FeatureSelectionTensorOps.RangeBinOneHot(Tensor<T>.FromMatrix(data), _nBins, _nBins), _nBins);
        var probabilities = engine.TensorDivideScalar(counts, NumOps.FromDouble(n));

        // Compute Renyi entropy: H_alpha(X) = 1/(1-alpha) * log2(sum(p^alpha)) over bins with p above 1e-10
        var shape = new[] { p, _nBins };
        var sumPowAlpha = engine.ReduceSum(engine.TensorWhere(engine.TensorGreaterThan(probabilities, floor),
            engine.TensorPower(probabilities, NumOps.FromDouble(_alpha)), new Tensor<T>(shape)), new[] { 1 }, false);
        var usable = engine.TensorGreaterThan(sumPowAlpha, floor);
        var safeSum = engine.TensorWhere(usable, sumPowAlpha, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One));
        var renyiEntropies = engine.TensorWhere(usable,
            engine.TensorDivideScalar(engine.TensorLog2(safeSum), NumOps.FromDouble(1 - _alpha)), new Tensor<T>(new[] { p }));
        _renyiEntropies = renyiEntropies.ToVector();
        // Select features with highest entropy (most informative)
        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(renyiEntropies, numToSelect);

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        Fit(data);
        return Transform(data);
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RenyiEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("RenyiEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("RenyiEntropySelector has not been fitted.");

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
