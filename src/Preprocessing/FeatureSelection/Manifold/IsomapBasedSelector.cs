using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Isomap-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Uses Isomap (Isometric Mapping) concepts to identify features that
/// best preserve geodesic distances along the data manifold.
/// </para>
/// <para><b>For Beginners:</b> Isomap measures distances along the surface
/// of your data (like measuring distance along a curved road rather than
/// cutting through). Features that help preserve these "along the surface"
/// distances capture the true structure of your data.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IsomapBasedSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private Vector<T>? _featureScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public Vector<T>? FeatureScores => _featureScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public IsomapBasedSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 10,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nFeaturesToSelect < 1)
            throw new ArgumentException("Number of features must be at least 1.", nameof(nFeaturesToSelect));

        _nFeaturesToSelect = nFeaturesToSelect;
        _nNeighbors = nNeighbors;
    }

    /// <inheritdoc />
    protected override void FitCore(Matrix<T> data)
    {
        _nInputFeatures = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        var engine = Engine;
        var x = Tensor<T>.FromMatrix(data);
        int k = Math.Min(_nNeighbors, n - 1);

        // Geodesic distances: Floyd-Warshall shortest paths through the symmetrized k-NN graph
        var euclidean = FeatureSelectionTensorOps.PairwiseDistances(x);
        var neighbors = FeatureSelectionTensorOps.KNearestMask(euclidean, k);
        var adjacency = engine.TensorMax(neighbors, engine.TensorTranspose(neighbors));
        var geodesic = FeatureSelectionTensorOps.ShortestPaths(euclidean, adjacency, out var reachable);

        // Feature importance: mean over the connected pairs of (single-feature distance x geodesic distance)
        var pairs = engine.TensorMultiply(reachable, FeatureSelectionTensorOps.OneMinus(engine.TensorEye<T>(n)));
        T pairCount = engine.TensorSum(pairs);
        var scores = new Tensor<T>(new[] { p });
        if (NumOps.GreaterThan(pairCount, NumOps.Zero))
        {
            var weighted = FeatureSelectionTensorOps.AbsDifferenceSums(x, engine.TensorMultiply(pairs, geodesic));
            scores = engine.TensorDivideScalar(engine.Reshape(weighted, new[] { p }), pairCount);
        }

        // Normalize scores
        if (p > 0)
        {
            T maxScore = engine.TensorMaxValue(scores);
            if (NumOps.GreaterThan(maxScore, NumOps.Zero))
                scores = engine.TensorDivideScalar(scores, maxScore);
        }

        _featureScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

        IsFitted = true;
    }

    /// <inheritdoc />
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapBasedSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IsomapBasedSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapBasedSelector has not been fitted.");

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
