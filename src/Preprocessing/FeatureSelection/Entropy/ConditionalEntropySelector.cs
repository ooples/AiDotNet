using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Conditional Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their ability to reduce uncertainty about the target,
/// measured by conditional entropy H(Y|X).
/// </para>
/// <para><b>For Beginners:</b> Conditional entropy measures how much uncertainty
/// remains about the target after knowing a feature's value. Features that reduce
/// uncertainty the most (lower conditional entropy) are most informative.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class ConditionalEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _entropyReductions;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? EntropyReductions => _entropyReductions;
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

        var engine = Engine;
        T count = NumOps.FromDouble(n);

        // Discretize target and features: bin = min(bins - 1, trunc((v - min) / range * bins)), 0 when constant
        var yBins = FeatureSelectionTensorOps.RangeBinOneHot(
            engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins, _nBins);
        var xBins = FeatureSelectionTensorOps.RangeBinOneHot(Tensor<T>.FromMatrix(data), _nBins, _nBins);

        // Compute H(Y)
        T hY = engine.TensorSum(FeatureSelectionTensorOps.EntropyBits(
            FeatureSelectionTensorOps.BinCounts(yBins, _nBins), n));

        // Compute H(Y|X) = -sum p(x, y) log2 p(y | x) over the non-empty joint cells
        var joint = FeatureSelectionTensorOps.JointBinCounts(xBins, yBins, _nBins);
        var cellShape = new[] { p, _nBins, _nBins };
        var xCounts = engine.TensorBroadcastTo(engine.Reshape(
            FeatureSelectionTensorOps.BinCounts(xBins, _nBins), new[] { p, _nBins, 1 }), cellShape);
        var occupied = engine.TensorGreaterThan(joint, NumOps.Zero);
        var safeXCounts = engine.TensorWhere(occupied, xCounts, PreprocessingTensorOps.Filled(cellShape, NumOps.One));
        var conditional = engine.TensorDivide(joint, safeXCounts);
        var safeConditional = engine.TensorWhere(occupied, conditional, PreprocessingTensorOps.Filled(cellShape, NumOps.One));
        var terms = engine.TensorMultiply(engine.TensorDivideScalar(joint, count), engine.TensorLog(safeConditional));
        var hYGivenX = engine.TensorDivideScalar(engine.TensorNegate(engine.ReduceSum(terms, new[] { 1, 2 }, false)),
            NumOps.FromDouble(Math.Log(2)));

        // Information gain: I(Y;X) = H(Y) - H(Y|X)
        var entropyReductions = engine.TensorAddScalar(engine.TensorNegate(hYGivenX), hY);
        _entropyReductions = entropyReductions.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(entropyReductions, numToSelect);

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
