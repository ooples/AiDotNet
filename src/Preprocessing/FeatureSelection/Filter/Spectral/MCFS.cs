using AiDotNet.Helpers;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.FeatureSelection.Filter.Spectral;

/// <summary>
/// Multi-Cluster Feature Selection using spectral graph analysis.
/// </summary>
/// <remarks>
/// <para>
/// MCFS uses spectral analysis to find features that preserve the multi-cluster
/// structure in data. It computes cluster indicators via spectral clustering
/// and then selects features using sparse regression with L1 regularization.
/// </para>
/// <para><b>For Beginners:</b> MCFS finds features that help preserve the natural
/// groupings (clusters) in your data. It first discovers hidden cluster structures
/// using spectral methods, then picks features that best capture those clusters.
/// Great when you don't have labels but want features that preserve data structure.
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations.</typeparam>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class MCFS<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nClusters;
    private readonly int _nNeighbors;
    private readonly double _alpha;
    private readonly int? _randomState;

    private Vector<T>? _mcfsScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? MCFSScores => _mcfsScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public MCFS(
        int nFeaturesToSelect = 10,
        int nClusters = 5,
        int nNeighbors = 5,
        double alpha = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));
        if (nClusters < 2)
            throw new ArgumentException("Number of clusters must be at least 2.", nameof(nClusters));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nClusters = nClusters;
        _nNeighbors = nNeighbors;
        _alpha = alpha;
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

        // k-NN affinity graph with heat-kernel weights exp(-d^2 / (2 sigma_i^2)), sigma_i the mean distance from
        // instance i to its k neighbours (1 when that is 0); symmetrised with the larger weight
        var distances = engine.TensorCDist(x, x, 2.0);
        var neighbours = FeatureSelectionTensorOps.KnnMask(distances, _nNeighbors);
        var neighbourCount = engine.ReduceSum(neighbours, new[] { 1 }, false);
        var sigma = FilterStatisticTensorOps.GuardedDivide(
            engine.ReduceSum(engine.TensorMultiply(distances, neighbours), new[] { 1 }, false), neighbourCount, 0.0, NumOps.Zero);
        sigma = engine.TensorWhere(engine.TensorLessThan(sigma, NumOps.FromDouble(1e-10)), FilterStatisticTensorOps.FilledLike(sigma, NumOps.One), sigma);
        var bandwidth = engine.TensorMultiplyScalar(engine.TensorSquare(sigma), NumOps.FromDouble(2.0));
        var directed = engine.TensorMultiply(
            engine.TensorExp(engine.TensorNegate(engine.TensorDivide(
                engine.TensorSquare(distances), FilterStatisticTensorOps.ColumnBroadcast(bandwidth, n)))),
            neighbours);
        var affinity = engine.TensorMax(directed, engine.TensorTranspose(directed));

        var propagation = Propagation(affinity);

        // Bottom k eigenvectors of L (cluster indicators): power iteration on I - L from random starts
        var rand = _randomState.HasValue
            ? RandomHelper.CreateSeededRandom(_randomState.Value)
            : RandomHelper.CreateSecureRandom();
        var starts = FeatureSelectionTensorOps.CenteredUniformStarts<T>(rand, _nClusters, n);
        var indicators = engine.TensorConcatenate(
            FeatureSelectionTensorOps.PowerIteration(propagation, starts, 100, keepPreviousOnZeroNorm: false).ToArray(), 1);

        // Per feature and cluster indicator the regression coefficient cov / var, soft-thresholded by alpha / n;
        // score = the norm of those coefficients over the clusters
        var centered = FilterStatisticTensorOps.CenterColumns(x);
        var centeredIndicators = FilterStatisticTensorOps.CenterColumns(indicators);
        var covariance = engine.TensorMatMul(engine.TensorTranspose(centered), centeredIndicators);
        var variance = engine.ReduceSum(engine.TensorSquare(centered), new[] { 0 }, false);
        var coefficient = engine.TensorAbs(engine.TensorDivide(
            covariance, FilterStatisticTensorOps.ColumnBroadcast(
                engine.TensorWhere(engine.TensorGreaterThan(variance, NumOps.FromDouble(1e-10)), variance,
                    FilterStatisticTensorOps.FilledLike(variance, NumOps.One)), _nClusters)));
        var sparse = engine.TensorMax(engine.TensorSubtractScalar(coefficient, NumOps.FromDouble(_alpha / n)), NumOps.Zero);
        var informative = FilterStatisticTensorOps.ColumnBroadcast(
            engine.TensorGreaterThan(variance, NumOps.FromDouble(1e-10)), _nClusters);
        var scores = engine.TensorSqrt(engine.ReduceSum(
            engine.TensorSquare(engine.TensorMultiply(sparse, informative)), new[] { 1 }, false));
        _mcfsScores = scores.ToVector();

        // Select top features
        _selectedIndices = PreprocessingTensorOps.TopKIndicesSorted(scores, Math.Min(_nFeaturesToSelect, p));

        IsFitted = true;
    }

    public new Matrix<T> FitTransform(Matrix<T> data)
    {
        FitCore(data);
        return Transform(data);
    }

    /// <summary>
    /// I - L for the normalised Laplacian L = I - D^-1/2 A D^-1/2 of an affinity [n, n] (degrees offset by 1e-10),
    /// scaled by rows then columns: for an isolated sample the outer product of the two 1/sqrt(eps) factors
    /// overflows Half and 0 * inf gave NaN, and 1e-10 itself is 0 in Half.
    /// </summary>
    internal static Tensor<T> Propagation(Tensor<T> affinity)
    {
        var engine = AiDotNetEngine.Current;
        int n = affinity.Shape[0];
        var degreeScale = engine.TensorReciprocal(engine.TensorSqrt(
            engine.TensorAddScalar(engine.ReduceSum(affinity, new[] { 1 }, false), FeatureSelectionTensorOps.Tiny<T>(1e-10))));
        var rows = engine.TensorMultiply(
            affinity, engine.TensorBroadcastTo(engine.Reshape(degreeScale, new[] { n, 1 }), new[] { n, n }));
        return engine.TensorMultiply(
            rows, engine.TensorBroadcastTo(engine.Reshape(degreeScale, new[] { 1, n }), new[] { n, n }));
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCFS has not been fitted.");

        return FilterStatisticTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("MCFS does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("MCFS has not been fitted.");

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
