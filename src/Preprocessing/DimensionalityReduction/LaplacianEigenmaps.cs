using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Laplacian Eigenmaps for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Laplacian Eigenmaps constructs a weighted graph from the data and finds a low-dimensional
/// embedding that preserves local neighborhood relationships by minimizing a cost function
/// based on the graph Laplacian.
/// </para>
/// <para>
/// The algorithm:
/// 1. Constructs a k-nearest neighbor or epsilon-neighborhood graph
/// 2. Computes edge weights using a kernel (e.g., heat kernel)
/// 3. Computes the graph Laplacian: L = D - W
/// 4. Finds eigenvectors of the generalized eigenvalue problem: L*y = λ*D*y
/// </para>
/// <para><b>For Beginners:</b> Laplacian Eigenmaps finds a low-dimensional representation where:
/// - Connected points in the graph stay close together
/// - The embedding respects the local geometry of the data
/// - It's similar to spectral clustering but for dimensionality reduction
///
/// Use cases:
/// - Manifold learning when data lies on a curved surface
/// - Image segmentation and clustering
/// - When you want to preserve local connectivity
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class LaplacianEigenmaps<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly int _nNeighbors;
    private readonly double? _radius;
    private readonly LaplacianAffinityType _affinity;
    private readonly double _gamma;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private int _nSamples;

    /// <summary>
    /// Gets the number of components (dimensions).
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the number of neighbors.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="LaplacianEigenmaps{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="nNeighbors">Number of neighbors for graph construction. Defaults to 5.</param>
    /// <param name="radius">Radius for epsilon-neighborhood (if null, uses k-NN). Defaults to null.</param>
    /// <param name="affinity">Affinity type for edge weights. Defaults to NearestNeighbors.</param>
    /// <param name="gamma">Kernel coefficient for RBF affinity. Defaults to 1.0.</param>
    /// <param name="randomState">Kept for API compatibility; the eigen solver is deterministic.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public LaplacianEigenmaps(
        int nComponents = 2,
        int nNeighbors = 5,
        double? radius = null,
        LaplacianAffinityType affinity = LaplacianAffinityType.NearestNeighbors,
        double gamma = 1.0,
        int? randomState = null,
        int[]? columnIndices = null)
        : base(columnIndices)
    {
        if (nComponents < 1)
        {
            throw new ArgumentException("Number of components must be at least 1.", nameof(nComponents));
        }

        if (nNeighbors < 1)
        {
            throw new ArgumentException("Number of neighbors must be at least 1.", nameof(nNeighbors));
        }

        _nComponents = nComponents;
        _nNeighbors = nNeighbors;
        _radius = radius;
        _affinity = affinity;
        _gamma = gamma;
    }

    /// <summary>
    /// Fits Laplacian Eigenmaps and computes the embedding.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        var engine = Engine;
        _nSamples = data.Rows;
        int n = data.Rows;

        // Step 1: Compute affinity matrix (weighted adjacency)
        var w = ComputeAffinityMatrix(Tensor<T>.FromMatrix(data), n);

        // Step 2: Degrees D_i = sum_j W_ij
        var degree = DimensionalityReductionOps.RowSums(w);

        // Step 3: Normalized Laplacian L = I - D^(-1/2) W D^(-1/2), with L_ij = 0 where W_ij = 0
        var zero = NumOps.Zero;
        var degreeProduct = engine.TensorMatMul(
            DimensionalityReductionOps.AsColumn(degree), DimensionalityReductionOps.AsRow(degree));
        var positiveProduct = engine.TensorGreaterThan(degreeProduct, zero);
        var safeRoot = engine.TensorSqrt(engine.TensorWhere(
            positiveProduct, degreeProduct, DimensionalityReductionOps.Full(new[] { n, n }, NumOps.One)));
        var connected = engine.TensorMultiply(engine.TensorGreaterThan(w, zero), positiveProduct);
        var laplacian = engine.TensorWhere(
            connected,
            engine.TensorNegate(engine.TensorDivide(w, safeRoot)),
            DimensionalityReductionOps.Full(new[] { n, n }, zero));
        engine.TensorFillDiagonalInPlace(laplacian, NumOps.One);

        // Step 4: Eigenvectors of L in ascending eigenvalue order
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(laplacian, descending: false);

        // Step 5: Select eigenvectors k + 1 (skipping the trivial one), falling back to k past the end
        var columns = new int[_nComponents];
        for (int k = 0; k < _nComponents; k++)
        {
            columns[k] = k + 1 < n ? k + 1 : k;
        }

        _embedding = DimensionalityReductionOps.SelectColumns(
            eigenvectors, DimensionalityReductionOps.IndexTensor(columns)).ToMatrix();
    }

    private Tensor<T> ComputeAffinityMatrix(Tensor<T> x, int n)
    {
        var engine = Engine;
        var squared = DimensionalityReductionOps.SquaredDistances(x, x);
        var negGamma = NumOps.FromDouble(-_gamma);

        if (_affinity == LaplacianAffinityType.RBF)
        {
            // Full RBF kernel with a zero diagonal
            var kernel = engine.TensorExp(engine.TensorMultiplyScalar(squared, negGamma));
            engine.TensorFillDiagonalInPlace(kernel, NumOps.Zero);
            return kernel;
        }

        // k-NN connectivity: W_ij = 1 (or exp(-gamma d^2)) for the k nearest j of i, then max-symmetrized
        var distances = engine.TensorSqrt(squared);
        var neighbors = DimensionalityReductionOps.NearestNeighbors(distances, _nNeighbors);
        int take = neighbors.Shape[1];
        var weights = _affinity == LaplacianAffinityType.NearestNeighbors
            ? DimensionalityReductionOps.Full(new[] { n, take }, NumOps.One)
            : engine.TensorExp(engine.TensorMultiplyScalar(
                DimensionalityReductionOps.TakeRows(squared, neighbors), negGamma));
        var w = DimensionalityReductionOps.ScatterRows(neighbors, weights, n, NumOps.Zero);
        return engine.TensorMax(w, engine.TensorTranspose(w));
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("LaplacianEigenmaps has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "LaplacianEigenmaps does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("LaplacianEigenmaps does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"LE{i + 1}";
        }
        return names;
    }
}
