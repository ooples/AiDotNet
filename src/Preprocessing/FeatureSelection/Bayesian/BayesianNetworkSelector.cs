using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Bayesian;

/// <summary>
/// Bayesian Network structure-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features based on their connections in a learned Bayesian network
/// structure, identifying features directly connected to the target.
/// </para>
/// <para><b>For Beginners:</b> A Bayesian network shows how variables depend on
/// each other. This selector finds features that have direct probabilistic
/// relationships with the target - these are the features that matter most
/// for prediction.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class BayesianNetworkSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, ISupervisedFeatureSelector<T>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _maxParents;

    private Vector<T>? _connectionStrengths;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int MaxParents => _maxParents;
    public Vector<T>? ConnectionStrengths => _connectionStrengths;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public BayesianNetworkSelector(
        int nFeaturesToSelect = 10,
        int maxParents = 3,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _maxParents = maxParents;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        throw new InvalidOperationException(
            "BayesianNetworkSelector requires target values. Use Fit(Matrix<T> data, Vector<T> target) instead.");
    }

    // Upper bound on the elements of one block of the pairwise joint histogram ([p * nBins, block * nBins]).
    private const int PairBlockElements = 1 << 22;

    /// <inheritdoc />
    public void Fit(Matrix<T> data, Vector<T> target)
    {
        if (data.Rows != target.Length)
            throw new ArgumentException("Target length must match rows in data.");

        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        int nBins = Math.Max(5, (int)Math.Sqrt(n / 5.0));
        var X = Tensor<T>.FromMatrix(data);
        var y = engine.Reshape(Tensor<T>.FromVector(target), new[] { n, 1 });
        var xBins = PreprocessingTensorOps.EqualWidthBinOneHot(X, nBins, out var xVarying);
        var yBins = PreprocessingTensorOps.EqualWidthBinOneHot(y, nBins, out var yVarying);

        // Mutual information between each feature and the target ([p])
        var miWithTarget = engine.TensorMultiply(
            engine.Reshape(PreprocessingTensorOps.HistogramMutualInformation(xBins, yBins, nBins), new[] { p }),
            engine.TensorMultiply(xVarying, engine.TensorBroadcastTo(yVarying, new[] { p })));

        // Conditional mutual information (Markov blanket approximation):
        // redundancy(j, k) = min(MI(j; y), MI(k; y), MI(j; k)) over k != j, maximised over k. The pairwise MI runs
        // over bounded blocks of k: all pairs at once is a [p * nBins, p * nBins] joint histogram (3.2 GB of doubles
        // for 2000 features in 10 bins). Constant columns carry no information.
        int blockWidth = Math.Max(1, PairBlockElements / Math.Max(1, p * nBins * nBins));
        var maxRedundancy = new Tensor<T>(new[] { p });
        for (int start = 0; start < p; start += blockWidth)
        {
            int width = Math.Min(blockWidth, p - start);
            var blockBins = engine.TensorSlice(xBins, new[] { 0, start * nBins }, new[] { n, width * nBins });
            var blockVarying = engine.TensorSlice(xVarying, new[] { start }, new[] { width });
            var blockTarget = engine.TensorSlice(miWithTarget, new[] { start }, new[] { width });
            var miPairs = engine.TensorMultiply(
                PreprocessingTensorOps.HistogramMutualInformation(xBins, blockBins, nBins),
                engine.TensorOuter(xVarying, blockVarying));
            var redundancy = engine.TensorMin(
                engine.TensorMin(
                    engine.TensorBroadcastTo(engine.Reshape(miWithTarget, new[] { p, 1 }), new[] { p, width }),
                    engine.TensorBroadcastTo(engine.Reshape(blockTarget, new[] { 1, width }), new[] { p, width })),
                miPairs);

            // k = start + c is feature j itself on row start + c of column c
            var self = engine.TensorTranspose(engine.TensorOneHot<T>(
                new Tensor<int>(Enumerable.Range(start, width).ToArray(), new[] { width }), p));
            var offDiagonal = PreprocessingTensorOps.Not(self);
            maxRedundancy = engine.TensorMax(maxRedundancy,
                engine.ReduceMax(engine.TensorMultiply(redundancy, offDiagonal), new[] { 1 }, false));
        }

        // Connection strength = direct MI minus redundancy
        var strengths = engine.TensorMax(
            engine.TensorSubtract(miWithTarget, engine.TensorMultiplyScalar(maxRedundancy, NumOps.FromDouble(0.5))),
            NumOps.Zero);
        _connectionStrengths = strengths.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(strengths, numToSelect);

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
            throw new InvalidOperationException("BayesianNetworkSelector has not been fitted.");

        return PreprocessingTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("BayesianNetworkSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("BayesianNetworkSelector has not been fitted.");

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
