using AiDotNet.Helpers;
using AiDotNet.Tensors.LinearAlgebra;
using AiDotNet.Enums;
using AiDotNet.Attributes;

namespace AiDotNet.Preprocessing.DimensionalityReduction;

/// <summary>
/// Spectral Embedding for nonlinear dimensionality reduction.
/// </summary>
/// <remarks>
/// <para>
/// Spectral Embedding forms an affinity matrix from the data and computes
/// the eigenvectors of the graph Laplacian. This provides a low-dimensional
/// representation that preserves local connectivity.
/// </para>
/// <para>
/// The algorithm constructs a similarity graph and uses spectral decomposition
/// of the Laplacian matrix to find coordinates that respect graph structure.
/// </para>
/// <para><b>For Beginners:</b> Spectral Embedding uses graph theory:
/// - Build a graph where similar points are connected
/// - Use the graph's structure to find good coordinates
/// - Similar to what's used in spectral clustering
/// - Good for data with cluster structure
/// </para>
/// </remarks>
/// <typeparam name="T">The numeric type for calculations (e.g., float, double).</typeparam>
[ComponentType(ComponentType.DimensionReducer)]
[PipelineStage(PipelineStage.Preprocessing)]
public class SpectralEmbedding<T> : TransformerBase<T, Matrix<T>, Matrix<T>>
{
    private readonly int _nComponents;
    private readonly SpectralAffinity _affinity;
    private readonly double _gamma;
    private readonly int _nNeighbors;

    // Fitted parameters
    private Matrix<T>? _embedding;
    private Matrix<T>? _affinityMatrix;
    private int _nSamples;
    private int _nFeaturesIn;

    /// <summary>
    /// Gets the number of components.
    /// </summary>
    public int NComponents => _nComponents;

    /// <summary>
    /// Gets the affinity type.
    /// </summary>
    public SpectralAffinity Affinity => _affinity;

    /// <summary>
    /// Gets the gamma parameter for RBF kernel.
    /// </summary>
    public double Gamma => _gamma;

    /// <summary>
    /// Gets the number of neighbors for nearest neighbors affinity.
    /// </summary>
    public int NNeighbors => _nNeighbors;

    /// <summary>
    /// Gets the affinity matrix.
    /// </summary>
    public Matrix<T>? AffinityMatrix => _affinityMatrix;

    /// <summary>
    /// Gets the embedding result.
    /// </summary>
    public Matrix<T>? Embedding => _embedding;

    /// <summary>
    /// Gets whether this transformer supports inverse transformation.
    /// </summary>
    public override bool SupportsInverseTransform => false;

    /// <summary>
    /// Creates a new instance of <see cref="SpectralEmbedding{T}"/>.
    /// </summary>
    /// <param name="nComponents">Target dimensionality. Defaults to 2.</param>
    /// <param name="affinity">How to construct the affinity matrix. Defaults to NearestNeighbors.</param>
    /// <param name="gamma">Kernel coefficient for RBF. If null, uses 1/n_features.</param>
    /// <param name="nNeighbors">Number of neighbors for NN affinity. Defaults to 10.</param>
    /// <param name="randomState">Kept for API compatibility; the eigen solver is deterministic.</param>
    /// <param name="columnIndices">The column indices to use, or null for all columns.</param>
    public SpectralEmbedding(
        int nComponents = 2,
        SpectralAffinity affinity = SpectralAffinity.NearestNeighbors,
        double? gamma = null,
        int nNeighbors = 10,
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
        _affinity = affinity;
        _gamma = gamma ?? 0; // Will be computed during fit
        _nNeighbors = nNeighbors;
    }

    /// <summary>
    /// Fits Spectral Embedding by computing the graph Laplacian eigenvectors.
    /// </summary>
    protected override void FitCore(Matrix<T> data)
    {
        _nSamples = data.Rows;
        _nFeaturesIn = data.Columns;
        int n = data.Rows;
        int p = data.Columns;

        // Compute gamma if not provided
        T gamma = NumOps.FromDouble(_gamma > 0 ? _gamma : 1.0 / p);

        // Step 1: Compute affinity matrix
        var affinity = ComputeAffinityMatrix(Tensor<T>.FromMatrix(data), n, gamma);
        _affinityMatrix = affinity.ToMatrix();

        // Step 2: Compute normalized graph Laplacian
        var laplacian = ComputeNormalizedLaplacian(affinity, n);

        // Step 3: Find smallest eigenvectors of Laplacian
        _embedding = ComputeEmbedding(laplacian, n).ToMatrix();
    }

    private Tensor<T> ComputeAffinityMatrix(Tensor<T> x, int n, T gamma)
    {
        var engine = Engine;
        var negGamma = NumOps.Negate(gamma);
        Tensor<T> affinity;

        switch (_affinity)
        {
            case SpectralAffinity.RBF:
                // Radial Basis Function (Gaussian) kernel
                affinity = engine.TensorExp(engine.TensorMultiplyScalar(
                    DimensionalityReductionOps.SquaredDistances(x, x), negGamma));
                break;

            case SpectralAffinity.NearestNeighbors:
                // Symmetric kNN graph: exp(-gamma d^2) where j is a neighbor of i or i a neighbor of j
                var distances = DimensionalityReductionOps.EuclideanDistances(x, x);
                var neighbors = DimensionalityReductionOps.NearestNeighbors(distances, _nNeighbors);
                var neighborDistances = DimensionalityReductionOps.TakeRows(distances, neighbors);
                var weights = engine.TensorExp(engine.TensorMultiplyScalar(engine.TensorSquare(neighborDistances), negGamma));
                var directed = DimensionalityReductionOps.ScatterRows(neighbors, weights, n, NumOps.Zero);
                affinity = engine.TensorMax(directed, engine.TensorTranspose(directed));
                break;

            case SpectralAffinity.Precomputed:
                throw new InvalidOperationException(
                    "Precomputed affinity requires passing the affinity matrix directly.");

            default:
                affinity = DimensionalityReductionOps.Full(new[] { n, n }, NumOps.Zero);
                break;
        }

        // Set diagonal to 1
        engine.TensorFillDiagonalInPlace(affinity, NumOps.One);
        return affinity;
    }

    private Tensor<T> ComputeNormalizedLaplacian(Tensor<T> w, int n)
    {
        var engine = Engine;

        // D^(-1/2), zero where the degree is at most 1e-10
        var degree = DimensionalityReductionOps.RowSums(w);
        var positive = engine.TensorGreaterThan(degree, NumOps.FromDouble(1e-10));
        var safeDegree = engine.TensorWhere(positive, degree, DimensionalityReductionOps.Full(n, NumOps.One));
        var dInv = engine.TensorWhere(
            positive, engine.TensorRsqrt(safeDegree), DimensionalityReductionOps.Full(n, NumOps.Zero));

        // L = I - D^(-1/2) W D^(-1/2)
        var normalized = DimensionalityReductionOps.ScaleColumns(DimensionalityReductionOps.ScaleRows(w, dInv), dInv);
        return engine.TensorSubtract(DimensionalityReductionOps.Identity<T>(n), normalized);
    }

    private Tensor<T> ComputeEmbedding(Tensor<T> laplacian, int n)
    {
        // Eigenvectors of L in ascending eigenvalue order (ties in index order)
        var (_, eigenvectors) = DimensionalityReductionOps.SymmetricEigen(laplacian, descending: false);

        // Take eigenvectors 1 to nComponents (skip the first which is constant)
        int nComp = Math.Min(_nComponents, n - 1);
        return Engine.TensorNarrow(eigenvectors, 1, 1, nComp);
    }

    /// <summary>
    /// Returns the embedding computed during Fit.
    /// </summary>
    protected override Matrix<T> TransformCore(Matrix<T> data)
    {
        if (_embedding is null)
        {
            throw new InvalidOperationException("SpectralEmbedding has not been fitted.");
        }

        if (data.Rows != _nSamples)
        {
            throw new InvalidOperationException(
                "SpectralEmbedding does not support out-of-sample transformation. " +
                "Use FitTransform() on the complete dataset.");
        }

        return _embedding.Clone();
    }

    /// <summary>
    /// Inverse transformation is not supported.
    /// </summary>
    protected override Matrix<T> InverseTransformCore(Matrix<T> data)
    {
        throw new NotSupportedException("SpectralEmbedding does not support inverse transformation.");
    }

    /// <summary>
    /// Gets the output feature names after transformation.
    /// </summary>
    public override string[] GetFeatureNamesOut(string[]? inputFeatureNames = null)
    {
        var names = new string[_nComponents];
        for (int i = 0; i < _nComponents; i++)
        {
            names[i] = $"Spectral{i + 1}";
        }
        return names;
    }
}

/// <summary>
/// Specifies how to construct the affinity matrix for Spectral Embedding.
/// </summary>
public enum SpectralAffinity
{
    /// <summary>
    /// K-nearest neighbors graph.
    /// </summary>
    NearestNeighbors,

    /// <summary>
    /// Radial Basis Function (Gaussian) kernel.
    /// </summary>
    RBF,

    /// <summary>
    /// Use a precomputed affinity matrix.
    /// </summary>
    Precomputed
}
