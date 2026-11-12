using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Unsupervised;

/// <summary>
/// Maximal Information feature selection based on entropy for unsupervised learning.
/// </summary>
/// <remarks>
/// <para>
/// Maximal Information selects features with highest entropy (information content).
/// Features with higher entropy contain more information and can better capture
/// the underlying structure of the data.
/// </para>
/// <para><b>For Beginners:</b> Entropy measures how "surprising" or "uncertain"
/// a feature's values are. A feature that's always the same is boring (low entropy).
/// A feature with lots of different values is informative (high entropy). This method
/// selects the most informative features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MaximalInformation<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _entropies;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? Entropies => _entropies;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MaximalInformation(
        int nFeaturesToSelect = 10,
        int nBins = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nBins < 2)
            throw new ArgumentException("Number of bins must be at least 2.", nameof(nBins));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nBins = nBins;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Discretize every column into nBins equal-width bins over [min, max] (numpy.histogram: the bin is
        // floor((x - min) / (max - min) * nBins), the maximum falling in the last bin); a constant column
        // falls entirely in bin 0 and so has zero entropy
        var x = Tensor<T>.FromMatrix(data);
        var min = Engine.Reshape(PreprocessingTensorOps.ColumnMin(x), new[] { 1, p });
        var range = Engine.TensorSubtract(Engine.Reshape(PreprocessingTensorOps.ColumnMax(x), new[] { 1, p }), min);
        var varying = PreprocessingTensorOps.AtLeast(range, NumOps.FromDouble(1e-10));
        var safeRange = Engine.TensorWhere(varying, range, PreprocessingTensorOps.Filled(new[] { 1, p }, NumOps.One));
        var scaled = Engine.TensorMultiplyScalar(
            Engine.TensorDivide(Engine.TensorSubtract(x, min), safeRange), NumOps.FromDouble(_nBins));
        var binIndex = Engine.TensorMin(Engine.TensorFloor(scaled), NumOps.FromDouble(_nBins - 1));
        binIndex = Engine.TensorMultiply(binIndex, Engine.TensorBroadcastTo(varying, new[] { n, p }));

        // Compute entropy -sum p_b ln p_b over the non-empty bins
        // (bin counts summed as int: a 0/1 sum in T stops counting at 2048 for Half)
        var counts = Engine.ReduceSum(
            PreprocessingTensorOps.OneHotBins(binIndex, _nBins).Cast<int>(), new[] { 0 }, false).Cast<T>();
        var probabilities = Engine.TensorDivideScalar(counts, NumOps.FromDouble(n));
        var occupied = Engine.TensorGreaterThan(counts, NumOps.Zero);
        var safeProbabilities = Engine.TensorWhere(occupied, probabilities,
            PreprocessingTensorOps.Filled(new[] { p * _nBins }, NumOps.One));
        var terms = Engine.TensorMultiply(probabilities, Engine.TensorLog(safeProbabilities));
        var entropies = Engine.TensorNegate(Engine.ReduceSum(
            Engine.Reshape(terms, new[] { p, _nBins }), new[] { 1 }, false));
        _entropies = entropies.ToVector();

        // Select top features by entropy
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(entropies, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformation has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MaximalInformation does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MaximalInformation has not been fitted.");

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
