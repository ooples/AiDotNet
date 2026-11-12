using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Entropy;

/// <summary>
/// Joint Entropy based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their joint entropy with the target, favoring
/// features that share information with the target.
/// </para>
/// <para><b>For Beginners:</b> Joint entropy measures the total randomness when
/// considering a feature and target together. Lower joint entropy relative to
/// individual entropies indicates the feature and target are related.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class JointEntropySelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nBins;

    private Vector<T>? _normalizedMutualInfo;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NBins => _nBins;
    public Vector<T>? NormalizedMutualInfo => _normalizedMutualInfo;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public JointEntropySelector(
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
            "JointEntropySelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
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

        // Discretize target and features: bin = min(bins - 1, trunc((v - min) / range * bins)), 0 when constant
        var yBins = FeatureSelectionTensorOps.RangeBinOneHot(
            engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 }), _nBins, _nBins);
        var xBins = FeatureSelectionTensorOps.RangeBinOneHot(Tensor<T>.FromMatrix(data), _nBins, _nBins);

        // Compute H(Y), H(X) and the joint entropy H(X,Y) (bits)
        T hY = engine.TensorSum(FeatureSelectionTensorOps.EntropyBits(
            FeatureSelectionTensorOps.BinCounts(yBins, _nBins), n));
        var hX = FeatureSelectionTensorOps.EntropyBits(FeatureSelectionTensorOps.BinCounts(xBins, _nBins), n);
        var hXY = FeatureSelectionTensorOps.EntropyBits(engine.Reshape(
            FeatureSelectionTensorOps.JointBinCounts(xBins, yBins, _nBins), new[] { p, _nBins * _nBins }), n);

        // Mutual information I(X;Y) = H(X) + H(Y) - H(X,Y), normalized by the mean entropy (0 when not above 1e-10)
        var entropySum = engine.TensorAddScalar(hX, hY);
        var mi = engine.TensorSubtract(entropySum, hXY);
        var avgEntropy = engine.TensorDivideScalar(entropySum, NumOps.FromDouble(2));
        var usable = engine.TensorGreaterThan(avgEntropy, PreprocessingTensorOps.Tolerance<T>(1e-10));
        var normalizedMutualInfo = engine.TensorWhere(usable,
            engine.TensorDivide(mi, engine.TensorWhere(usable, avgEntropy, PreprocessingTensorOps.Filled(new[] { p }, NumOps.One))),
            new Tensor<T>(new[] { p }));
        _normalizedMutualInfo = normalizedMutualInfo.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(normalizedMutualInfo, numToSelect);

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
            throw new InvalidOperationException("JointEntropySelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("JointEntropySelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("JointEntropySelector has not been fitted.");

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
