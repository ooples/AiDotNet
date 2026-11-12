using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;
using AiDotNet.Preprocessing.FeatureSelection.Helpers;

namespace AiDotNet.Preprocessing.FeatureSelection.Manifold;

/// <summary>
/// Isomap-inspired Manifold-based Feature Selection.
/// </summary>
/// <remarks>
/// <para>
/// Selects features that best preserve geodesic distances in the data manifold,
/// identifying features important for the underlying geometric structure.
/// </para>
/// <para><b>For Beginners:</b> Data often lies on a curved surface (manifold).
/// This selector finds features that best preserve the true distances along
/// that surface, keeping features important for the data's intrinsic geometry.
/// </para>
/// </remarks>
[ComponentType(ComponentType.FeatureSelector)]
[PipelineStage(PipelineStage.Preprocessing)]
public class IsomapSelector<T> : TransformerBase<T, Matrix<T>, Matrix<T>>, IFixedCountFeatureSelector<T>
{
    private readonly int _nFeaturesToSelect;
    private readonly int _nNeighbors;

    private Vector<T>? _manifoldScores;
    private int[]? _selectedIndices;
    private int _nInputFeatures;

    /// <inheritdoc />
    public int NFeaturesToSelect => _nFeaturesToSelect;
    public int NNeighbors => _nNeighbors;
    public Vector<T>? ManifoldScores => _manifoldScores;
    /// <inheritdoc />
    public int[]? SelectedIndices => _selectedIndices;
    /// <inheritdoc />
    public override bool SupportsInverseTransform => false;

    public IsomapSelector(
        int nFeaturesToSelect = 10,
        int nNeighbors = 5,
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

        // Geodesic distances along the manifold: shortest paths through the symmetrized k-NN graph whose
        // edges are Euclidean lengths (the Isomap construction; scikit-learn's Isomap uses the same graph)
        var euclidean = FeatureSelectionTensorOps.PairwiseDistances(x);
        var neighbors = FeatureSelectionTensorOps.KNearestMask(euclidean, k);
        var adjacency = engine.TensorMax(neighbors, engine.TensorTranspose(neighbors));
        var geodesic = FeatureSelectionTensorOps.ShortestPaths(euclidean, adjacency, out var reachable);

        // Each feature scores the Pearson correlation between its single-feature distances |x_i - x_j| and
        // the geodesic distances, over the connected pairs i != j
        var pairs = engine.TensorMultiply(reachable, FeatureSelectionTensorOps.OneMinus(engine.TensorEye<T>(n)));
        T pairCount = engine.TensorSum(pairs);
        var scores = new Tensor<T>(new[] { p });
        if (NumOps.GreaterThan(pairCount, NumOps.Zero))
        {
            T meanGeodesic = NumOps.Divide(engine.TensorSum(engine.TensorMultiply(pairs, geodesic)), pairCount);
            var centeredGeodesic = engine.TensorMultiply(pairs, engine.TensorSubtractScalar(geodesic, meanGeodesic));
            T geodesicVariation = engine.TensorSum(engine.TensorSquare(centeredGeodesic));

            var sums = FeatureSelectionTensorOps.AbsDifferenceSums(x, pairs, centeredGeodesic); // [p, 2]
            var distanceSum = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 0 }, new[] { p, 1 }), new[] { p });
            var covariation = engine.Reshape(engine.TensorSlice(sums, new[] { 0, 1 }, new[] { p, 1 }), new[] { p });

            // sum_ij P_ij (x_i - x_j)^2 = 2 (sum_i deg_i x_i^2 - x' P x) for the symmetric pair mask P
            var degree = engine.ReduceSum(pairs, new[] { 1 }, true);
            var squaredSum = engine.TensorMultiplyScalar(
                engine.TensorSubtract(
                    engine.ReduceSum(engine.TensorMultiply(engine.TensorSquare(x), degree), new[] { 0 }, false),
                    engine.ReduceSum(engine.TensorMultiply(x, engine.TensorMatMul(pairs, x)), new[] { 0 }, false)),
                NumOps.FromDouble(2));
            var distanceVariation = engine.TensorMax(
                engine.TensorSubtract(squaredSum, engine.TensorDivideScalar(engine.TensorSquare(distanceSum), pairCount)),
                NumOps.Zero);

            if (NumOps.GreaterThan(geodesicVariation, NumOps.FromDouble(1e-10)))
            {
                var defined = engine.TensorGreaterThan(distanceVariation, NumOps.FromDouble(1e-10));
                var safeVariation = engine.TensorWhere(defined, distanceVariation, FeatureSelectionTensorOps.Filled(new[] { p }, NumOps.One));
                var correlation = engine.TensorDivide(
                    covariation, engine.TensorSqrt(engine.TensorMultiplyScalar(safeVariation, geodesicVariation)));
                scores = engine.TensorWhere(defined, correlation, scores);
            }
        }

        _manifoldScores = scores.ToVector();

        int numToSelect = Math.Min(_nFeaturesToSelect, p);
        _selectedIndices = FeatureSelectionTensorOps.TopKIndicesSorted(scores, numToSelect);

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
            throw new InvalidOperationException("IsomapSelector has not been fitted.");

        return FeatureSelectionTensorOps.SelectColumns(data, _selectedIndices);
    }

    /// <inheritdoc />
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("IsomapSelector does not support inverse transformation.");
    }

    /// <inheritdoc />
    public bool[] GetSupportMask()
    {
        if (_selectedIndices is null)
            throw new InvalidOperationException("IsomapSelector has not been fitted.");

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
