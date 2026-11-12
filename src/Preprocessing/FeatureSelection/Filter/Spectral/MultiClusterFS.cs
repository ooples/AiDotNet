using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Multi-Cluster Feature Selection (MCFS) using spectral analysis for unsupervised feature selection.
/// </summary>
/// <remarks>
/// <para>
/// MCFS uses spectral clustering to discover the underlying cluster structure of the data,
/// then selects features that best preserve this structure. It combines spectral embedding
/// with sparse regression to select informative features.
/// </para>
/// <para><b>For Beginners:</b> MCFS first discovers natural groups (clusters) in your data
/// without knowing the labels. Then it picks features that best explain these groupings.
/// If a feature helps distinguish between clusters, it's selected. This is useful when
/// you have no labels but want to find discriminative features.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MultiClusterFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClusters;
    private readonly int _nNeighbors;
    private readonly double _sigma;
    private readonly int? _randomState;

    private Vector<T>? _mcfsScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NClusters => _nClusters;
    public Vector<T>? McfsScores => _mcfsScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MultiClusterFS(
        int nFeaturesToSelect = 10,
        int nClusters = 5,
        int nNeighbors = 5,
        double sigma = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClusters = nClusters;
        _nNeighbors = nNeighbors;
        _sigma = sigma;
        _randomState = randomState;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;
        var engine = Engine;
        var x = FilterStatisticTensorOps.AsTensor(data);

        var random = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();

        // k-NN graph with Gaussian weights exp(-d^2 / (2 sigma^2)) and its normalised affinity D^-1/2 W D^-1/2
        var W = FeatureSelectionTensorOps.KnnGaussianAffinity(x, _nNeighbors, 1.0 / (2 * _sigma * _sigma));
        var D = engine.ReduceSum(W, new[] { 1 }, false);
        var dInvSqrt = FilterStatisticTensorOps.GuardedDivide(
            FilterStatisticTensorOps.FilledLike(D, NumOps.One), engine.TensorSqrt(D), Math.Sqrt(1e-10), NumOps.Zero);
        var L = engine.TensorMultiply(W, engine.TensorOuter(dInvSqrt, dInvSqrt));

        // Eigenvectors by power iteration from random starts
        int k = Math.Min(_nClusters, n - 1);
        T lower = NumOps.FromDouble(-0.5);
        T upper = NumOps.FromDouble(0.5);
        var starts = Enumerable.Range(0, Math.Max(k, 0))
            .Select(_ => engine.TensorRandomUniformRange<T>(new[] { n, 1 }, lower, upper, random.Next()))
            .ToList();
        var eigenvectors = FeatureSelectionTensorOps.PowerIteration(L, starts, 50, keepPreviousOnZeroNorm: true);

        // standardised features (population std); score = max_k |f' v_k| / n
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var std = engine.TensorSqrt(FilterStatisticTensorOps.ColumnPopulationVariance(x));
        var scale = FilterStatisticTensorOps.GuardedDivide(FilterStatisticTensorOps.FilledLike(std, NumOps.One), std, 1e-10, NumOps.One);
        var spread = FilterStatisticTensorOps.RowBroadcast(engine.TensorGreaterThan(std, NumOps.FromDouble(1e-10)), n);
        var standardized = engine.TensorWhere(
            spread, engine.TensorMultiply(centered, FilterStatisticTensorOps.RowBroadcast(scale, n)), x);
        var scores = eigenvectors.Count == 0
            ? new Tensor<T>(new[] { p })
            : engine.ReduceMax(engine.TensorDivideScalar(engine.TensorAbs(engine.TensorMatMul(
                engine.TensorTranspose(standardized), engine.TensorConcatenate(eigenvectors.ToArray(), 1))), NumOps.FromDouble(n)),
                new[] { 1 }, false);
        scores = engine.TensorMax(scores, NumOps.Zero);
        _mcfsScores = scores.ToVector();

        // Select features with highest MCFS scores
        _selectedIndices = FilterStatisticTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiClusterFS has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MultiClusterFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MultiClusterFS has not been fitted.");

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
