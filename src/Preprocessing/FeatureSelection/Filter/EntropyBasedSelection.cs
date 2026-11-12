using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter;

/// <summary>
/// Entropy-Based Feature Selection using discretized feature entropy.
/// </summary>
/// <remarks>
/// <para>
/// Measures the information content of each feature using Shannon entropy.
/// Features with higher entropy contain more information and discriminative power.
/// Features with very low entropy (constant or near-constant) are removed.
/// </para>
/// <para><b>For Beginners:</b> Entropy measures how "mixed up" or unpredictable
/// a feature's values are. A feature that's always the same value has zero
/// entropy and tells you nothing useful. Features with higher entropy have
/// more variety and can better distinguish between different samples.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class EntropyBasedSelection<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;
    private readonly double _minEntropyRatio;

    private Vector<T>? _entropyScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? EntropyScores => _entropyScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public EntropyBasedSelection(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        double minEntropyRatio = 0.1,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
        _minEntropyRatio = minEntropyRatio;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        double maxPossibleEntropy = Math.Log(_nBins);

        // Entropy (nats) of every feature's range-scaled bins; a constant feature falls in one bin (entropy 0)
        var binCounts = FeatureSelectionTensorOps.BinCounts(
            PreprocessingTensorOps.RangeScaledBinOneHot(Tensor<T>.FromMatrix(data), _nBins), _nBins);
        var entropyScores = engine.TensorMultiplyScalar(
            FeatureSelectionTensorOps.EntropyBits(binCounts, n), NumOps.FromDouble(Math.Log(2)));
        _entropyScores = entropyScores.ToVector();

        // Filter by minimum entropy ratio
        var candidates = new HashSet<int>(PreprocessingTensorOps.MaskPositions(
            PreprocessingTensorOps.AtLeast(entropyScores, NumOps.FromDouble(_minEntropyRatio * maxPossibleEntropy))));

        int numToSelect = Math.Min(_nFeaturesToSelect, candidates.Count);
        _selectedIndices = PreprocessingTensorOps.StableOrder(entropyScores, descending: true)
            .Where(j => candidates.Contains(j))
            .Take(numToSelect)
            .OrderBy(x => x)
            .ToArray();

        IsFitted = true;
    }
    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EntropyBasedSelection has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("EntropyBasedSelection does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("EntropyBasedSelection has not been fitted.");

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
